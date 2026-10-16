# Video workflow

See the [project README](../README.md) for installation, dataset layout, exact
CLI commands, checkpoint dependencies, results and reproducibility limitations.
Run video commands from this directory.

- `baseline.py` trains/evaluates a baseline, then always prunes/fine-tunes and
  exports the selected full pruned model. It has no baseline-only switch.
  Run quantization explicitly with `experiment.py quantization`.
- `experiment.py` provides saved-model evaluation, frame/pruning experiments,
  CPU PTQ export, single-threshold adaptive evaluation and distribution plots.
- `video.py` owns dataset sampling and the classifier; `utils.py` owns shared
  training, checkpoint, metrics, calibration and profiling helpers.
- `adaptive_video.py` records adaptive decisions and provides the battery frame
  budget used by the audio/video integration notebook.
- `plot_adaptive_frontiers.py` replots saved threshold sweeps, preserving the
  historical fvcore-to-THOP normalization.

The current CLI does not have `--sweep`, `--thresholds`, optional baseline
pruning, or variable-frame pruning. Historical results made with those features
are preserved; consult their metadata and the project README before reproducing.
