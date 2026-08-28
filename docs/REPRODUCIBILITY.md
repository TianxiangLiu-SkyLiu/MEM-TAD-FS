# Reproducibility protocol

## Paper model

- Memory: 16 channels x 64 temporal slots;
- Queries: 80 with reference widths `[0.02,0.06,0.18]` and allocation
  `[40,25,15]`;
- Transformer: two encoder layers and three iterative decoder layers;
- Matching: class + segment L1 + temporal-IoU Hungarian cost;
- Classification: quality-aware sigmoid focal targets;
- Detection score: localization quality x class probability;
- Post-processing: flattened query-class top-k followed by class-aware
  Gaussian Soft-NMS;
- Model selection: EMA validation Avg. mAP;
- Default seed: 3407.

## Evaluation

`--map-eval-scope global` is the paper protocol. Predictions from all
validation videos are pooled by class and ranked by score. A prediction is a
true positive only when it has the correct class, reaches the requested tIoU,
and matches an otherwise unmatched ground-truth instance from the same video.

The evaluator computes mAP at tIoU 0.3 through 0.9. The paper configurations
set:

```yaml
avg_map_iou_thresholds: [0.3, 0.4, 0.5, 0.6, 0.7]
```

Consequently, the paper's `Avg.` column is the arithmetic mean of the five
reported mAP values from 0.3 to 0.7. Precision and Recall are reported at tIoU
0.5 after confidence filtering and Soft-NMS; they are operating-point metrics,
not inputs to AP integration.

## Checkpoints and resolved configs

Always report results from `best_checkpoint.pt` together with the generated
`resolved_config.yml`. Child YAML files inherit recursively through `_base_`,
so the child file alone is not a complete experimental record. The best
checkpoint normally stores EMA weights when EMA evaluation is enabled.

## Determinism

The provided configs request deterministic training and fix Python, NumPy, and
PyTorch seeds. Some CUDA attention and grid-sampling kernels can remain
non-deterministic depending on the PyTorch/CUDA build. Record the Git commit,
resolved config, hardware, and software versions with every run.

## Evaluation-only score ablation

`det_eval_class_score_only.yml` must not be independently trained. Evaluate it
with the exact M5 checkpoint used for quality-times-class scoring so that only
the ranking score changes.
