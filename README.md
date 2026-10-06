# Big Data at the Edge: MammAlps video baseline

Install dependencies and train:

```bash
uv sync --locked
uv run main.py --num-frames 4 --batch-size 8 --epochs 3
```

The fixed paths in `main.py` are `data/mammalps_v1/benchmark_1/clips/` and
`data/mammalps_v1/benchmark_1/metadata/`. The metadata directory must contain `train.csv`, `val.csv`,
and `test.csv`, each with `video_path` (relative to the clips directory) and
`species` columns. Existing split assignments are preserved; all frames from
a row stay in that row's split. Ensure the metadata keeps videos separate
across splits. The class mapping comes from training; unseen validation/test
species produce an error.

Each clip yields four frames by default, chosen nearest to equally spaced
presentation timestamps from the first to last decoded frame. `--num-frames`
changes this for all splits. One frame selects the midpoint; short clips repeat
nearest frames. PyAV handles decoding, including variable frame rates. The
first read of a clip in each worker uses one timestamp pass and one image pass;
subsequent reads reuse selected indices. Only selected, preprocessed images
are retained, rather than full decoded videos. Missing or invalid clips fail
with their path instead of silently changing the dataset.

Pretrained MobileNet V3 Large preprocessing produces `[T, 3, 224, 224]` per
video, and loaders produce `[B, T, 3, 224, 224]`. The model processes `B*T`
frames with the same 2D backbone and averages frame logits to `[B, classes]`.
Loss and accuracy are computed per video. This baseline uses several views
but does not learn motion or temporal order. Four frames require roughly four
times the frame computation; batch size counts videos (default 8).

Training validates every epoch and saves the lowest-validation-loss model to
`checkpoints/baseline.pt`. That checkpoint is reloaded for the final test
report. It contains `model_state_dict`, `class_to_idx`, sampling/preprocessing
configuration, epoch, and validation metrics. `utils.load_checkpoint` accepts
both this format and older plain state dictionaries. Checkpoints are intended
for inference, not exact optimizer-state training resumption.

Other options: `--num-workers`, `--seed`, and `--checkpoint`. Training uses
the `fit()` defaults: learning rate `1e-3` and weight decay `1e-4`. Seeded initialization and data ordering aid reproducibility;
exact results can still vary by device/backend. Pretrained weights are always used
(and downloaded if needed). Single-video inference timing runs automatically
and excludes decoding and preprocessing.
Use `uv run main.py --help` for defaults.

Synthetic checks (create tiny videos in a temporary directory, never read
project data):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m unittest discover -s tests -v
```
