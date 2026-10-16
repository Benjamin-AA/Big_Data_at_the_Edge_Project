"""Train and evaluate a MammAlps baseline and a channel-pruned copy."""
import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V3_Large_Weights

from video import MammAlpsDataset, Classifier
from utils import (get_device, fit, evaluate, count_parameters, model_size_mb,
                   load_checkpoint, benchmark_latency, count_flops, clone_model,
                   training_clip_transform, training_sampler)


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
    parser.add_argument("--target", choices=["activity", "species"], default="activity")
    parser.add_argument("--checkpoint", help="Default: checkpoints/<target>_baseline.pt")
    parser.add_argument("--load-baseline", action="store_true",
                        help="Load --checkpoint and skip baseline training before pruning")
    parser.add_argument("--augment", action="store_true",
                        help="Apply mild, temporally consistent training clip augmentation")
    parser.add_argument("--oversample", action="store_true",
                        help="Boost rare training classes with inverse-square-root frequency sampling")
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = f"checkpoints/{args.target}_baseline.pt"
    if args.num_workers < 0:
        parser.error("num-workers must be nonnegative")
    return args



def run_model(device, model, args, loaders, weights, mapping, input_size, save_full_model=False):
    """Fit, reload the selected checkpoint, evaluate and profile one model."""
    print(f"Device: {device}; {args.target} classes: {mapping}; frames/video: {args.num_frames}")

    fit(model, loaders["train"], loaders["val"], device, epochs=args.epochs,
        checkpoint_path=args.checkpoint,
        checkpoint_metadata={"target": args.target, "class_to_idx": mapping, "config": vars(args),
                             "sampling": "equidistant_timestamps_nearest_frame",
                             "preprocessing": str(weights), "aggregation": "mean_logits"})
    load_checkpoint(model, args.checkpoint, map_location=device)
    test_loss, test_acc = evaluate(model, loaders["test"], device)

    print(f"Best validation checkpoint: {args.checkpoint}")
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_acc:.2f}%")
    print(f"Parameters: {count_parameters(model):,}; model size: {model_size_mb(model):.2f} MB")

    flops = count_flops(
        model,
        input_size=input_size,
        device=device,
    )
    print(f"FLOPs/video: {flops / 1e9:.3f} GFLOPs")
    latency = benchmark_latency(model, input_size=input_size, device=device)
    print(f"Inference latency: {latency:.2f} ms/video (excludes decoding/preprocessing)")
    if save_full_model:
        checkpoint_path = Path(args.checkpoint)
        full_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_full{checkpoint_path.suffix}")
        saved_model = clone_model(model).cpu()
        saved_model.device = torch.device("cpu")
        torch.save({"model": saved_model, "target": args.target, "class_to_idx": mapping,
                    "num_frames": args.num_frames, "preprocessing": str(weights),
                    "aggregation": "mean_logits"}, full_path)
        print(f"Complete model: {full_path}")
    return model


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
            transform=weights.transforms(), target=args.target,
            clip_transform=training_clip_transform() if mode == "train" and args.augment else None,
        )
    sampler = training_sampler(datasets["train"], args.seed) if args.oversample else None
    loaders = {
        mode: DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=mode == "train" and sampler is None,
                         sampler=sampler if mode == "train" else None,
                         num_workers=args.num_workers, pin_memory=device.type == "cuda",
                         persistent_workers=args.num_workers > 0,
                         generator=torch.Generator().manual_seed(args.seed))
        for mode, dataset in datasets.items()
    }
    mapping = datasets["train"].class_to_idx
    input_size = (1, args.num_frames, 3, 224, 224)

    model_mobilenet_v3 = Classifier(device, input_size, None if args.load_baseline else weights, len(mapping))

    if args.load_baseline:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if "class_to_idx" in checkpoint and checkpoint["class_to_idx"] != mapping:
            raise ValueError("Baseline checkpoint class mapping does not match the dataset")
        if checkpoint.get("target", args.target) != args.target:
            raise ValueError("Baseline checkpoint target does not match --target")
        model_mobilenet_v3.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        baseline_model = model_mobilenet_v3.eval()
        print(f"Loaded baseline checkpoint: {args.checkpoint}")
    else:
        print("Baseline model run \n")
        baseline_model = run_model(model=model_mobilenet_v3, device=device, args=args, loaders=loaders, weights=weights, mapping=mapping, input_size=input_size)
    

    model_structured_depgraph = clone_model(baseline_model)
    model_structured_depgraph = model_structured_depgraph.to(device)
    model_structured_depgraph.prune_model()
    baseline_path = Path(args.checkpoint)
    pruned_path = baseline_path.with_name(f"{baseline_path.stem}_pruned{baseline_path.suffix}")
    pruned_args = argparse.Namespace(**vars(args))
    pruned_args.checkpoint = str(pruned_path)
    print("Pruned model run \n")
    run_model(model=model_structured_depgraph, device=device, args=pruned_args, loaders=loaders, weights=weights, mapping=mapping, input_size=input_size, save_full_model=True)


if __name__ == "__main__":
    main()
