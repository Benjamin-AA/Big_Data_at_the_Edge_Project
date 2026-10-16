"""Combine saved adaptive sweeps, with per-model Pareto lines and threshold labels.

Run from any directory: uv run Video/plot_adaptive_frontiers.py (from project root).
Legacy fvcore sweeps are normalized by profiling their saved checkpoint with THOP.
No dataset evaluation/training is performed. Only load trusted local checkpoints.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

VIDEO_DIR = Path(__file__).resolve().parent
ROOT = VIDEO_DIR / 'results' / 'adaptive_video'
DEFAULT_SWEEPS = [ROOT / name / 'threshold_sweep' for name in (
    'activity_augmented_test_16frames_cuda',
    'prune_0p1_full_test_16frames_cuda',
    'prune_0p25_full_test_16frames_cuda',
)]
DEFAULT_LABELS = ['Unpruned augmented baseline', '10% channel-pruned model', '25% channel-pruned model']
COLORS = ['#2563eb', '#d97706', '#059669', '#9333ea']
MARKERS = ['o', 's', '^', 'D']


def frame_cost(metadata):
    """Reuse THOP costs or normalize legacy fvcore runs from their checkpoint."""
    profiling = metadata.get('profiling', {})
    if profiling.get('profiler') == 'THOP' and profiling.get('multiplier') == 2:
        return float(metadata['per_frame_gflops'])
    if 'fvcore' not in metadata.get('compute_definition', ''):
        raise ValueError('Unknown profiling convention; cannot combine safely')
    import torch
    from video import Classifier
    from thop import profile
    checkpoint = Path(metadata['checkpoint'])
    if not checkpoint.is_file():
        checkpoint = VIDEO_DIR / 'checkpoints' / checkpoint.name
    torch.set_num_threads(1)
    full = checkpoint.stem.endswith('_full')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=not full)
    if full:
        model = saved['model'].cpu().eval()
    else:
        model = Classifier(torch.device('cpu'), (1, 1, 3, 224, 224), None, len(saved['class_to_idx']))
        model.load_state_dict(saved['model_state_dict'])
    with torch.inference_mode():
        macs, _ = profile(model.model.eval(), inputs=(torch.zeros(1, 3, 224, 224),), verbose=False)
    cost = 2 * macs / 1e9
    print(f'Normalized {checkpoint.name}: {cost:.9f} GFLOPs/frame')
    return cost


def frontier(points):
    """Keep nondominated points, minimizing compute and maximizing quality."""
    return sorted(set((x, y) for x, y in points if not any(
        xx <= x and yy >= y and (xx < x or yy > y) for xx, yy in points)))


def place_labels(ax, fig, labels, points, legend):
    """Choose nearby text positions, avoiding labels/markers, with leader lines."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = [legend.get_window_extent(renderer).expanded(1.02, 1.05)]
    point_boxes = []
    for point in points:
        x, y = ax.transData.transform(point)
        point_boxes.append(Bbox.from_bounds(x - 5, y - 5, 10, 10))
    for x, y, label, color in labels:
        px, py = ax.transData.transform((x, y))
        text = ax.text(x, y, label, fontsize=9, color=color, ha='center', va='center', zorder=6,
                       bbox=dict(facecolor='white', alpha=.85, edgecolor='none', pad=.6))
        best = None
        for dy in (18, -18, 32, -32, 48, -48, 65, -65, 85, -85):
            for dx in (0, 20, -20, 40, -40, 65, -65, 90, -90, 115, -115):
                pos = ax.transData.inverted().transform((px + dx, py + dy))
                text.set_position(pos)
                box = text.get_window_extent(renderer).expanded(1.10, 1.20)
                score = (100000 * sum(box.overlaps(b) for b in occupied)
                         + 1000 * sum(box.overlaps(b) for b in point_boxes) + dx * dx + dy * dy)
                if not ax.bbox.contains(box.x0, box.y0) or not ax.bbox.contains(box.x1, box.y1):
                    score += 1000000
                if best is None or score < best[0]:
                    best = (score, pos, box)
        text.set_position(best[1])
        occupied.append(best[2])
        ax.annotate('', xy=(x, y), xytext=best[1],
                    arrowprops=dict(arrowstyle='-', color=color, alpha=.5, lw=.7), zorder=2)


def plot(series, metric, title, output):
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=.10, right=.97, bottom=.13, top=.84)
    fig.text(.10, .95, f'Adaptive video: {title}–compute Pareto frontiers', fontsize=17, weight='bold', ha='left')
    metadata = series[0]['metadata']
    fig.text(.10, .905, f"{metadata['split'].capitalize()} set · maximum {metadata['max_frames']} frames per video · labels show confidence thresholds",
             fontsize=11, color='#64748b', ha='left')
    labels, points = [], []
    for i, item in enumerate(series):
        color, marker = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
        rows, cost = item['rows'], item['gflops_per_frame']
        values = [(float(r['average_frames']) * cost, float(r[metric])) for r in rows]
        front = frontier(values)
        ax.plot([x for x, _ in front], [y for _, y in front], color=color, marker=marker,
                linewidth=2, markersize=5, label=item['label'])
        # Always show static references and threshold 1, even when dominated.
        references = []
        for r, (x, y) in zip(rows, values):
            is_static = r['policy'] == 'static'
            is_one = not is_static and float(r['threshold']) == 1.0
            if is_static or is_one:
                references.append((x, y))
                ax.scatter(x, y, marker='*' if is_static else marker,
                           s=170 if is_static else 45, color=color,
                           edgecolors='white', linewidths=.7, zorder=5 if is_static else 4)
        visible = sorted(set(front + references))
        points.extend(visible)
        for x, y in visible:
            names = []
            for r, value in zip(rows, values):
                if value != (x, y):
                    continue

                if r['policy'] == 'static':
                    names.append('static')
                else:
                    threshold = float(r['threshold'])
                    if threshold == 0 or threshold > 0.5:
                        names.append(f'{threshold:g}')

            if names:
                labels.append((x, y, ', '.join(names), color))
    ax.scatter([], [], marker='*', s=150, color='#555555', label=f"Static {metadata['max_frames']}-frame references")
    ax.set(xlabel='Average GFLOPs per video', ylabel=f'{title} (%)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=.18)
    ax.margins(x=.10, y=.18)
    legend = ax.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='#e2e8f0')
    place_labels(ax, fig, labels, points, legend)
    stem = 'map' if metric == 'map_pct' else 'accuracy'
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'{stem}_pareto_three_models.{suffix}', dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweeps', nargs='+', type=Path, default=DEFAULT_SWEEPS)
    parser.add_argument('--labels', nargs='+', default=DEFAULT_LABELS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'combined_frontiers')
    args = parser.parse_args()
    if len(args.sweeps) != len(args.labels):
        parser.error('--sweeps and --labels must have the same number of entries')
    series = []
    for path, label in zip(args.sweeps, args.labels):
        metadata = json.loads((path / 'metadata.json').read_text())
        with (path / 'metrics.csv').open(newline='') as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f'No data in {path}')
        if series:
            for key in ('target', 'split', 'max_frames'):
                if metadata[key] != series[0]['metadata'][key]:
                    raise ValueError(f'Mismatched {key}: {path}')
            if {r['num_videos'] for r in rows} != {r['num_videos'] for r in series[0]['rows']}:
                raise ValueError('Sweeps evaluated different numbers of videos')
        series.append(dict(label=label, source=str(path.resolve()), metadata=metadata,
                           rows=rows, gflops_per_frame=frame_cost(metadata)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for metric, title in [('accuracy_pct', 'Accuracy'), ('map_pct', 'mAP')]:
        plot(series, metric, title, args.output_dir)
    (args.output_dir / 'three_models_provenance.json').write_text(json.dumps(dict(
        convention='2 * THOP MACs/frame times saved average frames; backbone estimates only',
        note='Per-model frontiers; source data unchanged; duplicate coordinates share threshold labels.',
        series=[{k: v for k, v in item.items() if k != 'rows'} for item in series]), indent=2))
    print(f'Saved accuracy and mAP plots to {args.output_dir}')


if __name__ == '__main__':
    main()
