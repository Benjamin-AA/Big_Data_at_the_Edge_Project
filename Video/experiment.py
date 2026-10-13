"""Video experiments: frame-count sweep, pruning sweep, and saved-model evaluation."""
import argparse
import csv
import json
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_data_arguments(parser):
    parser.add_argument('--batch-size', type=positive_int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--clips-dir', type=Path, default=Path('data/mammalps_v1/benchmark_1/clips'))
    parser.add_argument('--metadata-dir', type=Path, default=Path('data/mammalps_v1/benchmark_1/metadata'))


class Experiment:
    """Load one source checkpoint and share its mapping, preprocessing and data.

    Methods run frame sweeps, pruning sweeps, and saved-model evaluation.
    fresh_model() returns an independent copy, never the shared source model.
    """
    def __init__(self, args):
        import random
        import torch
        from torchvision.models import MobileNet_V3_Large_Weights
        from video import Classifier

        self.args = args
        self.seed = getattr(args, 'seed', 42)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.device = self.checked_device(args.device)
        self.weights = MobileNet_V3_Large_Weights.DEFAULT
        self.checkpoint_path = Path(args.checkpoint)
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu',
                                weights_only=not getattr(args, 'full_model', False))
        self.mapping = checkpoint.get('class_to_idx')
        if not self.mapping or sorted(self.mapping.values()) != list(range(len(self.mapping))):
            raise ValueError('Checkpoint needs a contiguous class_to_idx mapping')
        if checkpoint.get('target', 'activity') != 'activity':
            raise ValueError('Expected an activity checkpoint')
        if checkpoint.get('aggregation', 'mean_logits') != 'mean_logits':
            raise ValueError('Expected mean_logits aggregation')
        if checkpoint.get('preprocessing', str(self.weights)) != str(self.weights):
            raise ValueError('Checkpoint preprocessing differs from MobileNetV3 default preprocessing')
        self.trained_frames = checkpoint.get('num_frames', checkpoint.get('config', {}).get('num_frames', 'unknown'))
        self.selected_epoch = checkpoint.get('epoch', 'unknown')
        if getattr(args, 'full_model', False):
            self.model = checkpoint['model']
        else:
            self.model = Classifier(torch.device('cpu'), self.input_shape(4), None, len(self.mapping))
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.cpu().eval()
        self.model.device = torch.device('cpu')
        self._datasets = {}

    @staticmethod
    def checked_device(name):
        import torch
        device = torch.device(name)
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; choose a CPU device')
        return device

    @staticmethod
    def input_shape(frames):
        return (1, frames, 3, 224, 224)

    def fresh_model(self):
        from utils import clone_model
        model = clone_model(self.model).to(self.device).eval()
        model.device = self.device
        return model

    def dataset(self, split, frames):
        from video import MammAlpsDataset
        key = (split, frames)
        if key not in self._datasets:
            dataset = MammAlpsDataset(str(self.args.clips_dir), str(self.args.metadata_dir), split,
                                     num_frames=frames, class_to_idx=self.mapping,
                                     transform=self.weights.transforms(),
                                     sampling="nested" if self.args.command == "adaptive" else "uniform")
            unseen = {row['activity'] for row in dataset.metadata} - set(self.mapping)
            if not len(dataset) or unseen:
                raise ValueError(f'Invalid {split} dataset: empty or unseen activities {unseen}')
            self._datasets[key] = dataset
        return self._datasets[key]

    def loader(self, split, frames):
        import torch
        from torch.utils.data import DataLoader
        return DataLoader(self.dataset(split, frames), batch_size=self.args.batch_size,
                          shuffle=split == 'train', num_workers=self.args.num_workers,
                          pin_memory=self.device.type == 'cuda',
                          generator=torch.Generator().manual_seed(self.seed))

    def profile(self, model, frames, device=None):
        import torch
        from fvcore.nn import FlopCountAnalysis
        from utils import benchmark_latency
        device = self.checked_device(device if device is not None else self.device)
        model.to(device).eval()
        model.device = device
        shape = self.input_shape(frames)
        with torch.no_grad():
            analysis = FlopCountAnalysis(model, torch.randn(*shape, device=device))
            gflops = analysis.total() / 1e9
        latency = benchmark_latency(model, input_size=shape, device=device,
                                    n_runs=self.args.latency_runs)
        return gflops, latency, dict(analysis.unsupported_ops())


    def run_frames(self):
        args = self.args
        import torch
        from utils import evaluate_metrics

        experiment = self
        device, mapping = experiment.device, experiment.mapping
        trained_frames = experiment.trained_frames
        model = experiment.fresh_model()
        output_dir = args.output_dir or Path("results/frame_sweep") / f"{args.checkpoint.stem}_{args.split}_{args.device}"
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "metrics.csv"
        details = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                   "checkpoint": str(args.checkpoint.resolve()), "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
                   "trained_num_frames": trained_frames, "torch_version": str(torch.__version__),
                   "cpu_threads": torch.get_num_threads(),
                   "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                   "class_to_idx": mapping, "unsupported_operators": {}}
        rows = []
        print(f"Fixed checkpoint trained with {trained_frames} frames; evaluating {args.split} on {device}", flush=True)
        for frames in sorted(set(args.frames)):
            print(f"Evaluating {frames} frames/video...", flush=True)
            loader = experiment.loader(args.split, frames)
            metrics, _ = evaluate_metrics(model, loader, device, mapping)
            loss, accuracy = metrics['loss'], 100 * metrics['accuracy']
            map_pct = 100 * metrics['map']
            gflops, latency, unsupported = experiment.profile(model, frames)
            details["unsupported_operators"][str(frames)] = unsupported
            rows.append({"checkpoint": args.checkpoint.name, "split": args.split, "device": args.device,
                         "trained_num_frames": trained_frames, "num_frames": frames, "num_videos": len(loader.dataset),
                         "loss": loss, "accuracy_pct": accuracy, "map_pct": map_pct, "map_scope": metrics["map_scope"],
                         "absent_classes": json.dumps(metrics["absent_classes"]), "fvcore_gflops": gflops, "latency_ms": latency})
            # Save each completed point so a later interrupted run still leaves useful results.
            with csv_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output_dir / "run_info.json").write_text(json.dumps(details, indent=2))
            print(f"{frames} frames: accuracy={accuracy:.2f}%, mAP={map_pct:.2f}%, loss={loss:.4f}, "
                  f"{gflops:.3f} GFLOPs, {latency:.2f} ms/video", flush=True)
            del loader
        plot_comparison(csv_path, output_dir, "frame")
        print(f"Saved results and plots to {output_dir}")

    def run_pruning(self):
        args = self.args
        import random
        import torch
        import torch_pruning as tp
        from utils import fit, evaluate_metrics, load_checkpoint, count_parameters, model_size_mb

        experiment = self
        device = experiment.device
        bench_device = experiment.checked_device(args.benchmark_device)
        mapping, weights = experiment.mapping, experiment.weights
        args.output_dir.mkdir(parents=True, exist_ok=True)
        info = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "torch_version": str(torch.__version__), "cpu_threads": torch.get_num_threads(),
                "source_checkpoint": str(args.checkpoint.resolve()),
                "source_mtime_ns": args.checkpoint.stat().st_mtime_ns, "unsupported_operators": {}}
        def loaders():
            return {split: experiment.loader(split, 4) for split in {'train', 'val', args.split}}

        rows = []
        csv_path = args.output_dir / "metrics.csv"
        def measure(model, label, pruning_ratio, selected_epoch):
            model.to(device).eval()
            metrics, _ = evaluate_metrics(model, experiment.loader(args.split, 4), device, mapping)
            loss, accuracy = metrics['loss'], 100 * metrics['accuracy']
            map_pct = 100 * metrics['map']
            gflops, latency, unsupported = experiment.profile(model, 4, bench_device)
            info['unsupported_operators'][label] = unsupported
            rows.append(dict(model=label, pruning_ratio=pruning_ratio, num_frames=4, split=args.split,
                             num_videos=len(experiment.dataset(args.split, 4)), selected_epoch=selected_epoch,
                             accuracy_pct=accuracy, map_pct=map_pct, map_scope=metrics["map_scope"],
                             absent_classes=json.dumps(metrics["absent_classes"]), loss=loss, parameters=count_parameters(model),
                             size_mb=model_size_mb(model), fvcore_gflops=gflops,
                             latency_ms=latency, benchmark_device=args.benchmark_device))
            with csv_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            (args.output_dir / 'run_info.json').write_text(json.dumps(info, indent=2))
            print(f"{label}: accuracy={accuracy:.2f}%, mAP={map_pct:.2f}%, {gflops:.3f} GFLOPs, {latency:.2f} ms/video", flush=True)

        torch.manual_seed(args.seed)
        measure(experiment.fresh_model(), 'original_baseline', 0, experiment.selected_epoch)
        for amount in sorted(set(args.ratios)):
            random.seed(args.seed); torch.manual_seed(args.seed)
            model = experiment.fresh_model()
            label = f"prune_{str(amount).replace('.', 'p')}"
            if amount:
                pruner = tp.pruner.MagnitudePruner(model.model,
                    example_inputs=torch.randn(1, 3, 224, 224, device=device),
                    importance=tp.importance.MagnitudeImportance(p=2), pruning_ratio=amount,
                    global_pruning=False, iterative_steps=1, ignored_layers=[model.model.classifier[-1]])
                pruner.step()
                del pruner
            # Reset after tracing so training RNG starts consistently for the control too.
            random.seed(args.seed); torch.manual_seed(args.seed)
            data = loaders()
            path = args.output_dir / f'{label}.pt'
            metadata = dict(target='activity', class_to_idx=mapping, num_frames=4,
                            pruning_ratio=amount, preprocessing=str(weights), aggregation='mean_logits')
            print(f"Fine-tuning {amount:.0%} channel removal for {args.epochs} epochs", flush=True)
            fit(model, data['train'], data['val'], device, epochs=args.epochs, lr=args.lr,
                checkpoint_path=path, checkpoint_metadata=metadata)
            load_checkpoint(model, path, map_location=device)
            selected_epoch = torch.load(path, map_location='cpu', weights_only=True)['epoch']
            model.cpu().eval(); model.device = torch.device('cpu')
            torch.save({**metadata, 'model': model}, args.output_dir / f'{label}_full.pt')
            measure(model, label, amount, selected_epoch)
            del model, data
        plot_comparison(csv_path, args.output_dir, "pruning")
        print(f"Saved checkpoints, CSV, metadata and plots to {args.output_dir}")

    def run_adaptive(self):
        from adaptive_video import adaptive_evaluate
        frames = self.args.num_frames or (self.trained_frames if self.trained_frames != 'unknown' else 4)
        output = self.args.output_dir or Path('results/adaptive_video') / f'{self.checkpoint_path.stem}_{self.args.split}_{frames}frames_{self.args.device}'
        adaptive_evaluate(self.fresh_model(), self.loader(self.args.split, frames), self.device,
                          threshold=self.args.threshold, output_dir=output)

    def run_evaluation(self):
        args = self.args
        from utils import evaluate_metrics

        experiment = self
        device, mapping = experiment.device, experiment.mapping
        frames = args.num_frames or (experiment.trained_frames if experiment.trained_frames != 'unknown' else 4)
        model = experiment.fresh_model()
        loader = experiment.loader(args.split, frames)
        print(f'Evaluating {args.checkpoint} on {args.split}: {len(loader.dataset)} videos, {frames} frames, {device}', flush=True)
        aggregate, per_class = evaluate_metrics(model, loader, device, mapping)
        aggregate.update(checkpoint=str(args.checkpoint.resolve()), split=args.split,
                         device=args.device, num_frames=frames, class_to_idx=mapping)
        output = args.output_dir or Path('results/evaluation') / f'{args.checkpoint.stem}_{args.split}_{frames}frames_{args.device}'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'metrics.json').write_text(json.dumps(aggregate, indent=2, allow_nan=False))
        with (output / 'per_class.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_class[0]))
            writer.writeheader(); writer.writerows(per_class)
        for label, key in [('Accuracy', 'accuracy'), ('mAP', 'map')]:
            print(f'{label}: {100 * aggregate[key]:.2f}%')
        print('mAP averages classes present in this split.')
        if aggregate['absent_classes']:
            print('Classes without positive examples:', ', '.join(aggregate['absent_classes']))
        print(f'Saved fractional scores (0–1) to {output}; rerunning overwrites these result files.')


def plot_comparison(csv_path, output_dir, kind):
    """Plot accuracy and (when available) mAP; old accuracy-only CSVs still work."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with Path(csv_path).open(newline='') as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError('Empty results CSV')
    is_frames = kind == 'frame'
    if is_frames:
        rows.sort(key=lambda row: int(row['num_frames']))
    first = rows[0]
    metrics = [('accuracy_pct', 'Accuracy (%)')]
    if all(row.get('map_pct') not in (None, '') for row in rows):
        metrics.append(('map_pct', 'mAP (%) — present classes'))
    x_axes = [('num_frames' if is_frames else 'pruning_ratio',
               'Frames per video' if is_frames else 'Channel removal target'),
              ('fvcore_gflops', 'Profiled GFLOPs/video (fvcore)'),
              ('latency_ms', 'Model latency (ms/video)')]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(15, 4 * len(metrics) + 1), squeeze=False)
    for metric_axes, (key, ylabel) in zip(axes, metrics):
        for ax, (xkey, xlabel) in zip(metric_axes, x_axes):
            if is_frames:
                ax.plot([float(r[xkey]) for r in rows], [float(r[key]) for r in rows], 'o-')
            else:
                ax.scatter(float(first[xkey]), float(first[key]), marker='*', s=160, label='Original baseline')
                ax.scatter([float(r[xkey]) for r in rows[1:]], [float(r[key]) for r in rows[1:]], color='tab:orange')
            for index, row in enumerate(rows):
                label = f"{row['num_frames']}f" if is_frames else ('baseline' if index == 0 else f"{float(row['pruning_ratio']):.0%}")
                ax.annotate(label, (float(row[xkey]), float(row[key])), xytext=(5, 6), textcoords='offset points')
            ax.set(xlabel=xlabel, ylabel=ylabel, ylim=(0, 100))
            ax.grid(alpha=0.3)
    if is_frames:
        for row_axes in axes:
            row_axes[0].set_xticks([int(row['num_frames']) for row in rows])
        title = f"{first['checkpoint']} | {first['split']} | {first['device']} | fixed weights"
    else:
        axes[0, 0].legend()
        title = f"Four frames | {first['split']} | latency on {first['benchmark_device']} | 0% = fine-tuning control"
    fig.suptitle(title)
    fig.text(0.5, 0.01, 'Latency excludes decoding/preprocessing. FLOPs omit unsupported operators; 1 multiply-add = 1 FLOP.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        fig.savefig(output_dir / f'{kind}_comparison.{extension}', dpi=180)
    plt.close(fig)


def pruning_ratio(value):
    value = float(value)
    if not 0 <= value < 1:
        raise argparse.ArgumentTypeError("pruning ratios must be in [0, 1)")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, description in (
        ('frames', 'Evaluate several frame counts with fixed weights'),
        ('pruning', 'Prune and fine-tune independent copies at four frames'),
        ('evaluate', 'Report accuracy and mAP for a saved model'),
        ('adaptive', 'Evaluate midpoint-first frame exits and save frames used'),
    ):
        sub = commands.add_parser(command, help=description, description=description)
        sub.add_argument('--checkpoint', type=Path, default=Path('checkpoints/activity_baseline.pt'))
        sub.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if command == 'pruning' else 'cpu')
        sub.add_argument('--split', choices=['val', 'test'], default='test' if command == 'evaluate' else 'val')
        sub.add_argument('--seed', type=int, default=42)
        sub.add_argument('--output-dir', type=Path,
                         default=Path('results/pruning_sweep_4frames') if command == 'pruning' else None)
        if command == 'adaptive':
            sub.add_argument('--threshold', type=float, default=0.9)
        add_data_arguments(sub)
        if command != 'pruning':
            sub.add_argument('--full-model', action='store_true', help='Load a trusted complete model, e.g. pruned *_full.pt')
        if command in ('frames', 'pruning'):
            sub.add_argument('--latency-runs', type=positive_int, default=50)
            sub.add_argument('--plot-only', type=Path, metavar='CSV', help='Replot saved results without loading a model')
        if command == 'frames':
            sub.add_argument('--frames', nargs='+', type=positive_int, default=[1, 2, 4, 8, 16])
        elif command == 'pruning':
            sub.add_argument('--ratios', nargs='+', type=pruning_ratio, default=[0, 0.1, 0.3, 0.5, 0.7],
                             help='Channel removal fractions; 0 is an extra-fine-tuning control')
            sub.add_argument('--epochs', type=positive_int, default=3)
            sub.add_argument('--lr', type=float, default=1e-3)
            sub.add_argument('--benchmark-device', choices=['cpu', 'cuda'], default='cpu')
        else:
            sub.add_argument('--num-frames', type=positive_int, help='Default: saved frame count, otherwise 4')
    args = parser.parse_args(argv)
    if args.command == 'adaptive' and not 0 <= args.threshold <= 1:
        parser.error('--threshold must be between 0 and 1')
    if args.num_workers < 0:
        parser.error('--num-workers must be nonnegative')
    if args.command == 'pruning' and not 0 < args.lr < float('inf'):
        parser.error('--lr must be positive and finite')
    return args


def main(argv=None):
    args = parse_args(argv)
    if getattr(args, 'plot_only', None):
        kind = 'frame' if args.command == 'frames' else 'pruning'
        plot_comparison(args.plot_only, args.output_dir or args.plot_only.parent, kind)
        return
    experiment = Experiment(args)
    if args.command == 'frames':
        experiment.run_frames()
    elif args.command == 'pruning':
        experiment.run_pruning()
    elif args.command == 'adaptive':
        experiment.run_adaptive()
    else:
        experiment.run_evaluation()


if __name__ == '__main__':
    main()
