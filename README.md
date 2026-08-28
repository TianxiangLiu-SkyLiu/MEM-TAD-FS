# MEM-TAD

Official PyTorch implementation of **MEM-TAD: Learning Observation and
Evolution Memory States for Fine-Grained Temporal Action Detection in
Continuous Tennis Videos**.

MEM-TAD processes a sequence of pre-extracted video features with two
fixed-capacity recurrent states. The observation-detail memory preserves local
category and boundary evidence, while the residual-driven event-evolution
memory provides a contextual prior for multi-scale temporal queries. A
Transformer decoder then iteratively refines segment boundaries, class scores,
and localization quality.

![MEM-TAD overview](assets/mem_tad_framework.png)

## Release scope

This repository contains the model, training and global-mAP evaluation code,
paper configurations, controlled ablations, prediction analysis tools, and
dataset-format documentation. It does not redistribute professional tennis
broadcasts, pre-extracted features, annotations, checkpoints, or local
experiment outputs.

## Repository layout

```text
configs/tennisnet/   Main SlowFast, I3D, C3D, and TSP configurations
configs/ablation/    M0--M7, G1, Q1, detector, and sensitivity controls
models/              Dual-memory modules and MEM-TAD detector
utils/               Training, validation, matching, NMS, and data loading
tools/               Cache generation and prediction-analysis utilities
docs/                Data format and reproducibility protocol
assets/              Architecture diagrams
```

## Installation

The reported environment used Python 3.9, PyTorch 2.8.0, and CUDA 12.9.

```bash
conda create -n mem-tad python=3.9 -y
conda activate mem-tad
pip install -r requirements.txt
```

SciPy is required for the fast Hungarian matcher. Without it, the code falls
back to a substantially slower Python implementation.

## Data preparation

Place TennisNet annotations and one feature file per video under the following
layout, or edit the corresponding YAML paths:

```text
data/TennisNet/
├── annotations/
│   ├── tennisnet_annotations_low.json
│   ├── tennisnet_annotations_middle.json
│   └── tennisnet_annotations_high.json
└── features/
    ├── slowfast/<video_id>.npy
    ├── i3d/<video_id>.npy
    ├── c3d/<video_id>.npy
    └── tsp/<video_id>.npy
```

Feature files may contain `[N,C,T,H,W]` or `[N,C,H,W]` arrays, where `N` is
the number of sequential feature blocks. See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)
for the annotation schema, normalized segments, and supported tensor layouts.

## Training

Train the paper model with SlowFast features on TennisNet-middle:

```bash
python main.py \
  --cfg configs/tennisnet/slowfast_middle.yml \
  --mode train \
  --device cuda:0
```

Low- and high-granularity configurations are `slowfast_low.yml` and
`slowfast_high.yml`. Equivalent I3D, C3D, and TSP configurations are provided
in the same directory. Adjust `batch_size`, data-loader workers, and prefetch
settings when using a smaller machine.

Training writes `last_checkpoint.pt`, `best_checkpoint.pt`,
`resolved_config.yml`, metrics, and memory diagnostics below `output_dir`.

## Evaluation

```bash
python main.py \
  --cfg configs/tennisnet/slowfast_middle.yml \
  --mode val \
  --resume /path/to/best_checkpoint.pt \
  --device cuda:0 \
  --map-eval-scope global
```

Export post-processed predictions for error analysis:

```bash
python main.py \
  --cfg configs/tennisnet/slowfast_middle.yml \
  --mode val \
  --resume /path/to/best_checkpoint.pt \
  --device cuda:0 \
  --map-eval-scope global \
  --predictions-out outputs/predictions/mem_tad_middle.json
```

The paper protocol pools all validation predictions by class before computing
AP. The reported Avg. mAP is the mean of mAP at tIoU 0.3, 0.4, 0.5, 0.6, and
0.7. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for scoring,
matching, NMS, and model-selection details.

## Ablations

The full paper model is:

```bash
python main.py \
  --cfg configs/ablation/core_m5_full.yml \
  --mode train \
  --device cuda:0
```

The strong controls are:

- `strong_g1_single_convgru.yml`: one ConvGRU state;
- `strong_q1_fifo_history.yml`: non-recurrent FIFO feature history;
- `strong_m7_absolute_dual_state.yml`: equal-parameter absolute-state write.

All public core controls use the final 16-channel x 64-step memory. The full
ablation index is in [configs/ablation/README.md](configs/ablation/README.md).

## License and data availability

The code is released under the [MIT License](LICENSE). Raw professional sports
broadcasts cannot be redistributed by this repository. Users must obtain video
content in accordance with the original copyright holders' licenses. TennisNet
metadata, fixed splits, normalized temporal intervals, class definitions, and
evaluation tools are intended to be released separately from the copyrighted
videos.

## Citation

The manuscript is under review. Please use the metadata in
[CITATION.cff](CITATION.cff) when citing this repository; publication metadata
will be updated after acceptance.
