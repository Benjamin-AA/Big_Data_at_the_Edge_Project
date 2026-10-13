"""Video datasets and a frame-averaged MobileNet classifier."""
from bisect import bisect_left
import csv
from pathlib import Path
import utils
import av
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
import torch_pruning as tp

from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


def nested_frame_positions(frames):
    """Prefixes: midpoint, start, end, quarter, three-quarter, then finer gaps."""
    positions = [0.5, 0.0, 1.0]
    denominator = 4
    while len(positions) < frames:
        positions.extend(i / denominator for i in range(1, denominator, 2))
        denominator *= 2
    return positions[:frames]


class MammAlpsDataset(Dataset):
    def __init__(self, data_path_str, metadata_path_str, mode, num_frames=4,
                 class_to_idx=None, transform=None, sampling="uniform"):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown mode: {mode}")
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")

        self.data_path = Path(data_path_str)
        self.mode = mode
        if sampling not in {"uniform", "nested"}:
            raise ValueError("sampling must be uniform or nested")
        self.sampling = sampling
        self.num_frames = num_frames
        self.transform = transform or MobileNet_V3_Large_Weights.DEFAULT.transforms()
        self._sample_indices = {}
        metadata_file = Path(metadata_path_str) / f"{mode}.csv"

        with metadata_file.open(newline="") as f:
            reader = csv.DictReader(f)
            self.metadata = list(reader)

        activity = sorted({row["activity"] for row in self.metadata})
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(activity)}
        self.class_to_idx = dict(class_to_idx)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        clip = self.sample_clip(self.data_path / row["video_path"], self.num_frames)
        return clip, torch.tensor(self.class_to_idx[row["activity"]], dtype=torch.long)

    def sample_clip(self, clip, frames):
        """Sample nearest frames to equally spaced times, including both ends.

        One frame requests the midpoint. Short clips repeat nearest frames.
        Cache only selected indices, never decoded videos. Timestamps are read
        once per clip/worker; a second pass decodes the selected RGB images.
        """
        if not isinstance(frames, int) or frames < 1:
            raise ValueError("frames must be a positive integer")
        key = (str(clip), frames, self.sampling)
        try:
            if key not in self._sample_indices:
                with av.open(str(clip)) as container:
                    times = [frame.time for frame in container.decode(video=0)]

                if self.sampling == "nested":
                    positions = nested_frame_positions(frames)
                else:
                    positions = [0.5] if frames == 1 else [i / (frames - 1) for i in range(frames)]
                targets = [times[0] + (times[-1] - times[0]) * position for position in positions]
                indices = []
                for target in targets:
                    right = min(bisect_left(times, target), len(times) - 1)
                    left = max(0, right - 1)
                    indices.append(min((left, right), key=lambda i: abs(times[i] - target)))
                self._sample_indices[key] = indices
            indices = self._sample_indices[key]
            wanted = set(indices)
            selected = {}
            with av.open(str(clip)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index in wanted:
                        rgb = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)
                        selected[index] = self.transform(rgb)
                    if index >= max(indices):
                        break
            if wanted != selected.keys():
                raise ValueError("Could not decode all requested frames")
            return torch.stack([selected[index] for index in indices])
        except Exception as exc:
            raise RuntimeError(f"Failed to sample video {clip}: {exc}") from exc


class Classifier(nn.Module):
    def __init__(self, device, input_size, weights, num_classes):
        super().__init__()
        self.input_size = input_size
        self.model = mobilenet_v3_large(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        self.device = device
        self.to(device)
        self.weights = weights
        self.num_classes = num_classes
        self.battery_level = 100

    def forward(self, clips, adaptive_threshold=None, return_frames_used=False):
        if adaptive_threshold is not None and not 0 <= adaptive_threshold <= 1:
            raise ValueError("adaptive_threshold must be between 0 and 1")
        if clips.ndim == 4:
            logits = self.model(clips)
            counts = torch.ones(clips.shape[0], dtype=torch.long, device=clips.device)
            return (logits, counts) if return_frames_used else logits
        if clips.ndim != 5 or clips.shape[1] < 1:
            raise ValueError("Expected nonempty [B,T,C,H,W] videos or [B,C,H,W] images")

        batch, max_frames, channels, height, width = clips.shape

        if adaptive_threshold is not None:
            active_idx = torch.arange(batch, device=clips.device)
            logits_sum = torch.zeros(batch, self.num_classes, device=clips.device)
            frames_used = torch.zeros(batch, dtype=torch.long, device=clips.device)

            previous_frames = 0
            target_frames = 1

            while target_frames <= max_frames and len(active_idx) > 0:
                new_frames = clips[active_idx, previous_frames:target_frames]
                n_active = len(active_idx)
                n_new = target_frames - previous_frames

                new_frames = new_frames.reshape(
                    n_active * n_new, channels, height, width
                )

                new_logits = self.model(new_frames).reshape(
                    n_active, n_new, self.num_classes
                )

                logits_sum[active_idx] += new_logits.sum(dim=1)
                frames_used[active_idx] += n_new

                mean_logits = (
                    logits_sum[active_idx]
                    / frames_used[active_idx].unsqueeze(1)
                )

                confidence = torch.softmax(
                    mean_logits, dim=1
                ).max(dim=1).values

                active_idx = active_idx[confidence < adaptive_threshold]

                if target_frames == max_frames:
                    break
                previous_frames = target_frames
                # Evaluate midpoint, then both endpoints, then double the budget.
                target_frames = min(max_frames, 3 if target_frames == 1 else target_frames * 2)

            logits = logits_sum / frames_used.unsqueeze(1)
            return (logits, frames_used) if return_frames_used else logits

        logits = self.model(
            clips.reshape(batch * max_frames, channels, height, width)
        )

        logits = logits.reshape(batch, max_frames, -1).mean(dim=1)
        counts = torch.full((batch,), max_frames, dtype=torch.long, device=clips.device)
        return (logits, counts) if return_frames_used else logits

    @torch.no_grad()
    def evaluate_model(self, inputs):
        was_training = self.training
        self.eval()
        try:
            return self(inputs)
        finally:
            self.train(was_training)

    def prune_model(self, type=None):
        # utils.apply_structured_pruning_torch()


        example_inputs = torch.randn(1, 3, 224, 224, device=self.device)

        ignored_layers = [self.model.classifier[-1]]

        importance = tp.importance.MagnitudeImportance(p=2)
        pruner = tp.pruner.MagnitudePruner(model=self.model, example_inputs=example_inputs, importance=importance, pruning_ratio=0.3, ignored_layers=ignored_layers)
        pruner.step()



    def quantize_weights(self, cpu_device, calibrate_loader, n_batches=20, seed=42):
        """Apply static PTQ in place, retaining the video aggregation wrapper."""
        cpu_device = torch.device(cpu_device)
        if cpu_device.type != "cpu":
            raise ValueError("Static quantization requires a CPU device")
        backend = next((name for name in ("x86", "fbgemm", "qnnpack")
                        if name in torch.backends.quantized.supported_engines), None)
        if backend is None:
            raise RuntimeError("No supported static quantization backend available")
        torch.backends.quantized.engine = backend
        self.to(cpu_device).eval()
        self.device = cpu_device
        example_inputs = (torch.randn(1, *self.input_size[-3:]),)
        self.model = prepare_fx(
            self.model, get_default_qconfig_mapping(backend), example_inputs,
        )
        # The wrapper flattens video batches into frames before the backbone.
        utils.calibrate(self, calibrate_loader, cpu_device, n_batches=n_batches, seed=seed)
        self.model = convert_fx(self.model)
        return self
