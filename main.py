"""Train and evaluate a video-level MammAlps baseline."""
import argparse
import random

import torch
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights

from video import MammAlpsDataset, Classifier
from utils import (get_device, fit, evaluate, count_parameters, model_size_mb,
                   load_checkpoint, benchmark_latency)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-frames", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=8, help="Videos per batch")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/baseline.pt")
    args = parser.parse_args()
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    weights = MobileNet_V3_Large_Weights.DEFAULT
    datasets = {}
    for mode in ("train", "val", "test"):
        datasets[mode] = MammAlpsDataset(
            "data/mammalps_v1/benchmark_1/clips/",
            "data/mammalps_v1/benchmark_1/metadata/", mode, num_frames=args.num_frames,
            class_to_idx=None if mode == "train" else datasets["train"].class_to_idx,
            transform=weights.transforms(),
        )
    loaders = {
        mode: DataLoader(dataset, batch_size=args.batch_size, shuffle=mode == "train",
                         num_workers=args.num_workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0,
                         generator=torch.Generator().manual_seed(args.seed))
        for mode, dataset in datasets.items()
    }
    mapping = datasets["train"].class_to_idx
    model = Classifier(device, weights, len(mapping))
    print(f"Device: {device}; classes: {mapping}; frames/video: {args.num_frames}")

    fit(model, loaders["train"], loaders["val"], device, epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        checkpoint_metadata={"class_to_idx": mapping, "config": vars(args),
                             "sampling": "equidistant_timestamps_nearest_frame",
                             "preprocessing": str(weights), "aggregation": "mean_logits"})
    load_checkpoint(model, args.checkpoint, map_location=device)
    test_loss, test_acc = evaluate(model, loaders["test"], device)

    print(f"Best validation checkpoint: {args.checkpoint}")
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_acc:.2f}%")
    print(f"Parameters: {count_parameters(model):,}; model size: {model_size_mb(model):.2f} MB")

    latency = benchmark_latency(model, input_size=(1, args.num_frames, 3, 224, 224), device=device)
    print(f"Inference latency: {latency:.2f} ms/video (excludes decoding/preprocessing)")


if __name__ == "__main__":
    main()
