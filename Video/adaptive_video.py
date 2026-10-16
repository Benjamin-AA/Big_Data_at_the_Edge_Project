"""Adaptive frame evaluation and per-video compute accounting."""
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import SequentialSampler
from utils import evaluate_metrics


def select_policy_frames(battery):
    if battery > 0.75:
        return 8
    elif battery > 0.5:
        return 4
    elif battery > 0.25:
        return 2
    else:
        return 1

@torch.inference_mode()
def adaptive_evaluate(model, loader, device, threshold=0.9,
                      output_dir=Path('results/adaptive_video'), return_metrics=False):
    """Evaluate nested clips and save frame counts; return loss and accuracy (%).

    Set return_metrics=True to return the aggregate metrics dictionary instead.

    The dataset must use nested sampling and a sequential, non-dropping loader so
    saved rows can be matched to video metadata. All candidate frames are decoded
    upfront; only backbone execution is skipped on early exit.
    """
    if getattr(loader.dataset, 'sampling', None) != 'nested':
        raise ValueError('Adaptive evaluation requires sampling="nested"')
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError('Use shuffle=False and drop_last=False to save matching video IDs')
    model.to(device).eval()
    records = []

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = model

        def forward(self, clips):
            logits, counts = self.backbone(clips, adaptive_threshold=threshold, return_frames_used=True)
            confidence, prediction = logits.softmax(1).max(1)
            records.extend(zip(counts.cpu().tolist(), prediction.cpu().tolist(), confidence.cpu().tolist()))
            return logits

    aggregate, per_class = evaluate_metrics(Recorder(), loader, device, loader.dataset.class_to_idx)
    target = loader.dataset.target
    mapping = loader.dataset.class_to_idx
    names = {i: label for label, i in mapping.items()}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'frames_used.csv').open('w', newline='') as file:
        fields = ['video_path', target, 'prediction', 'confidence', 'frames_used', 'max_frames', 'early_exit']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, (count, pred, conf) in zip(loader.dataset.metadata, records):
            writer.writerow(dict(video_path=row['video_path'], **{target: row[target]},
                                 prediction=names[pred], confidence=conf, frames_used=count,
                                 max_frames=loader.dataset.num_frames, early_exit=count < loader.dataset.num_frames))
    counts = [record[0] for record in records]
    aggregate.update(threshold=threshold, sampling='nested', max_frames=loader.dataset.num_frames,
                     average_frames=sum(counts) / len(counts),
                     frame_count_histogram={str(n): counts.count(n) for n in sorted(set(counts))},
                     split=loader.dataset.mode, class_to_idx=mapping, target=target)
    (output_dir / 'metrics.json').write_text(json.dumps(aggregate, indent=2, allow_nan=False))
    with (output_dir / 'per_class.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(per_class[0]))
        writer.writeheader(); writer.writerows(per_class)
    print(f"Accuracy: {100 * aggregate['accuracy']:.2f}%; mAP: {100 * aggregate['map']:.2f}%; "
          f"average frames: {aggregate['average_frames']:.2f}/{loader.dataset.num_frames}")
    print(f"Saved adaptive results to {output_dir}")
    if return_metrics:
        return aggregate
    return aggregate['loss'], 100 * aggregate['accuracy']
