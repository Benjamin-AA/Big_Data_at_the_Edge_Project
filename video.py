"""Video datasets and a frame-averaged MobileNet classifier."""
from bisect import bisect_left
import csv
from pathlib import Path

import av
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class MammAlpsDataset(Dataset):
    def __init__(self, data_path_str, metadata_path_str, mode, num_frames=4,
                 class_to_idx=None, transform=None):
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unknown mode: {mode}")
        if not isinstance(num_frames, int) or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")

        self.data_path = Path(data_path_str)
        self.mode = mode
        self.num_frames = num_frames
        self.transform = transform or MobileNet_V3_Large_Weights.DEFAULT.transforms()
        self._sample_indices = {}
        metadata_file = Path(metadata_path_str) / f"{mode}.csv"

        with metadata_file.open(newline="") as f:
            reader = csv.DictReader(f)
            self.metadata = list(reader)

        species = sorted({row["species"] for row in self.metadata})
        if class_to_idx is None:
            class_to_idx = {name: i for i, name in enumerate(species)}
        self.class_to_idx = dict(class_to_idx)


    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata[idx]
        clip = self.sample_clip(self.data_path / row["video_path"], self.num_frames)
        return clip, torch.tensor(self.class_to_idx[row["species"]], dtype=torch.long)

    def sample_clip(self, clip, frames):
        """Sample nearest frames to equally spaced times, including both ends.

        One frame requests the midpoint. Short clips repeat nearest frames.
        Cache only selected indices, never decoded videos. Timestamps are read
        once per clip/worker; a second pass decodes the selected RGB images.
        """
        if not isinstance(frames, int) or frames < 1:
            raise ValueError("frames must be a positive integer")
        key = (str(clip), frames)
        try:
            if key not in self._sample_indices:
                with av.open(str(clip)) as container:
                    times = [frame.time for frame in container.decode(video=0)]

                targets = ([(times[0] + times[-1]) / 2] if frames == 1 else
                           [times[0] + (times[-1] - times[0]) * i / (frames - 1)
                            for i in range(frames)])
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
    def __init__(self, device, weights, num_classes):
        super().__init__()
        self.model = mobilenet_v3_large(weights=weights)
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        self.to(device)

    def forward(self, clips):
        if clips.ndim == 4:
            return self.model(clips)
        if clips.ndim != 5 or clips.shape[1] < 1:
            raise ValueError("Expected [B,T,C,H,W] clips or [B,C,H,W] images")
        batch, frames, channels, height, width = clips.shape
        logits = self.model(clips.reshape(batch * frames, channels, height, width))
        return logits.reshape(batch, frames, -1).mean(dim=1)

    @torch.no_grad()
    def evaluate_model(self, inputs):
        was_training = self.training
        self.eval()
        try:
            return self(inputs)
        finally:
            self.train(was_training)
