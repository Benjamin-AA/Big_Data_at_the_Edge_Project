# MammAlps Adaptive Audio and Video Inference

This project compares video-only inference with two audio-gated video pipelines.

## Setup

```bash
uv sync
```

Use a compatible PyTorch and torchvision installation for the target CPU or GPU.

## 1. Audio

Audio notebooks are in `Audio/`. Data preparation notebooks are in `data/`.

### Required files for the final evaluation

```text
data/metadata/benchmark_2_audio_model_manifest.csv
data/audio_cache/full/<file_id>.flac

Audio/models/pretrained/Cnn14_mAP=0.431.pth
Audio/models/panns_cnn14_binary_head_2p0s.pt
Audio/models/panns_exit1_block2_2p0s.pt
Audio/models/panns_exit2_block4_2p0s.pt
Audio/results/panns_early_exit_policy_results_2p0s.csv
Audio/results/panns_early_exit_standalone_2p0s.csv
Audio/results/panns_audio_all_paths_flops_2p0s.csv

Audio/models/mfcc_gate/mfcc_lr_2p0s.joblib
Audio/results/mfcc_gate/mfcc_lr_rf_test_comparison.csv
```

If these files exist, audio training is not needed.

### Audio notebooks

| Notebook | Purpose |
| --- | --- |
| `data/full_exploration.ipynb` | Optional dataset exploration |
| `data/Audio_files.ipynb` | Creates the primary audio manifest |
| `Audio/Prepare_Audio_Data.ipynb` | Creates the model manifest and audio cache |
| `Audio/YAMNet_Baseline.ipynb` | Optional YAMNet baseline |
| `Audio/PANN_baseline.ipynb` | Trains the final PANNs binary head |
| `Audio/PANN_EarlyExit.ipynb` | Trains early-exit heads and saves policies |
| `Audio/PANNs_Compression_Experiments.ipynb` | Optional pruning and INT8 experiments |
| `Audio/PANNs_Complete_Audio_Comparison.ipynb` | Profiles PANNs paths and saves FLOPs results |
| `Audio/MFCC_LR_RF_Audio_Gate.ipynb` | Trains MFCC + LR and RF |

Run only the notebooks needed to create missing files. YAMNet and compression are not required for the final pipeline.

Audio checkpoints are stored in `Audio/models/`. Results are stored in `Audio/results/`.

## 2. Video

*To be completed.*

## 3. Final Audio and Video Evaluation

Run `Audio_Video_Pipeline_Comparison.ipynb` from the project root.

It compares:

- Video only
- PANNs early exit + video
- MFCC + Logistic Regression + video

### Additional required files

```text
Video/video.py
Video/utils.py
Audio/Video_checkpoints/Pruned_10/prune_0p1_full.pt
```

The pipeline uses the trained 10% pruned MobileNetV3 Large model.

### Evaluation settings

| Setting | Value |
| --- | --- |
| Dataset | MammAlps Benchmark II test split |
| Audio windows | 207 |
| Video confidence threshold | 0.90 |
| Simulated battery levels | 100%, 70%, 40%, 20% |
| Maximum video frames | 8, 4, 2, 1 |

Raw video files are cached in `data/video_cache/raw/`. Missing videos can be downloaded when `DOWNLOAD_MISSING_VIDEOS = True`. This can take time.

### Outputs

Saved in `results/pipeline_comparison/`:

```text
pipeline_summary.csv
pipeline_per_sample.csv
audio_predictions.csv
video_predictions.csv
run_info.json
presentation_figures/
```

The reported GOPs are estimated operation counts, not measured energy usage. Benchmark II does not provide aligned video activity labels for the pipeline windows.
