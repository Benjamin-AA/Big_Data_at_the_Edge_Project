# Video workflow

See the [project README](../README.md) for setup, verified commands, checkpoint
provenance and historical reproduction limits. Run CLI commands here so relative
checkpoint/output paths resolve as documented. Data defaults are repository-root
`Data/mammalps_v1/benchmark_1/{clips,metadata}`, independent of working directory.

- `baseline.py` trains/evaluates the baseline; `--lr` defaults to 1e-3.
  `--prune` optionally fine-tunes a fixed 30% pruned copy using the same
  augmentation/oversampling settings. Quantization is a separate experiment.
- `experiment.py` provides evaluation, frame sweeps, configurable-frame pruning,
  CPU PTQ export, single-threshold adaptive inference, restored threshold sweeps
  (`--sweep` / `--thresholds`), and distribution plots.
- `adaptive_video.py` records actual adaptive decisions; sweeps request aggregate
  metrics while ordinary callers keep the original return format.
- `video.py` and `utils.py` retain model, sampling, training, checkpoint and metric
  behavior. Static/frame/pruning FLOPs remain fvcore.
- `plot_adaptive_frontiers.py` reads saved sweeps and retains its historical
  fvcore-to-THOP normalization. Recovered sweeps profile only their frame backbone
  using the stash's THOP ×2 convention.

Recovery was selective; stash `11b831e` was not applied or dropped. No historical
result files were rewritten.
