# MEM-TAD v19 Ablation Configs

All configs inherit from `base_19_middle.yml`. Child values override the base,
and both training and validation resolve `_base_` recursively. Training writes
the fully merged configuration to `resolved_config.yml` in the output folder.

The shared paper protocol uses TennisNet-middle, SlowFast features, global AP,
80 queries, a 16-channel x 64-step memory, seed 3407, and the final MEM-TAD
set-prediction head. Avg. mAP is the mean over tIoU thresholds 0.3--0.7.

## Core memory ablations

| Config | Meaning |
| --- | --- |
| `core_m0_stateless_observation.yml` | Current-observation projection without recurrent shallow history; deep memory disabled. |
| `core_m1_shallow_only.yml` | Recurrent observation-detail memory only. |
| `core_m2_deep_shallow_input_concat.yml` | Deep memory reads the shallow state; shallow/deep are concatenated. |
| `core_m3_deep_residual_concat.yml` | Deep memory reads the shallow-state change; shallow/deep are concatenated. |
| `core_m4_deep_prior_no_aux.yml` | Residual deep memory and deep-prior refinement, without prior auxiliary loss. |
| `core_m5_full.yml` | Full v19 model. |
| `core_m6_deep_reset_each_clip.yml` | Full parameters, but deep state is reset at every clip. |

The base files use seed 3407. Additional seed files are included where those
runs were used in the accompanying experiments.

## Strong state-memory controls

| Config | Meaning |
| --- | --- |
| `strong_g1_single_convgru.yml` | One standard ConvGRU state supplies both decoder evidence and proposal context. |
| `strong_q1_fifo_history.yml` | A non-recurrent FIFO queue stores the latest 64 projected observations. |
| `strong_m7_absolute_dual_state.yml` | Equal-parameter dual state where deep memory reads the absolute shallow state instead of its residual. |

## Detection-head ablations

| Config | Single change from full model |
| --- | --- |
| `det_single_scale_reference.yml` | One reference width (`0.06`). |
| `det_no_iterative_refine.yml` | Disable decoder iterative refinement. |
| `det_no_aux_decoder.yml` | Disable intermediate decoder supervision. |
| `det_no_denoising.yml` | Disable denoising queries and losses. |
| `det_eval_class_score_only.yml` | Evaluate with class probability only. |
| `det_no_ema.yml` | Train and evaluate without EMA. |

`det_eval_class_score_only.yml` is evaluation-only. It must load the exact
checkpoint used for the full quality-times-class evaluation.

## Sensitivity configs

- Memory shape: `sens_memory_c{16,32,64}_t64.yml` and
  `sens_memory_c32_t{32,64,128}.yml`.
- Query count: `sens_queries_q{42,60,80,100}.yml`.
- Deep-prior samples: `sens_prior_points_p{3,5,7,9}.yml`.
- Deep-prior context: `sens_prior_context_s{1p5,3p0,4p5}.yml`.

Only one axis changes in every sensitivity config. The default sensitivity
seed is 3407.

## Commands

Train one configuration:

```bash
python main.py \
  --cfg ./configs/ablation/core_m5_full.yml \
  --mode train \
  --device cuda:0
```

Evaluate the same full checkpoint with class score only:

```bash
python main.py \
  --cfg ./configs/ablation/det_eval_class_score_only.yml \
  --mode val \
  --resume /path/to/core_m5_full/seed3407/best_checkpoint.pt \
  --device cuda:0
```

Do not infer experimental settings from folder names. Use the saved
`resolved_config.yml`, checkpoint tensor shapes, and Git commit for reporting.
