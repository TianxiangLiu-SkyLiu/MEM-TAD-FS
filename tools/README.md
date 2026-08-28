# Tools

## Prediction diagnostics and bootstrap

First export each MEM-TAD prediction file with `main.py --mode val
--predictions-out ...`. Then run duration/category diagnostics and paired
per-video bootstrap:

```bash
python tools/analyze_tennisnet_predictions.py \
  --annotations data/TennisNet/annotations/tennisnet_annotations_middle.json \
  --predictions M5=outputs/predictions/m5.json \
                M1=outputs/predictions/m1.json \
                Q1=outputs/predictions/q1.json \
                M7=outputs/predictions/m7.json \
  --main-model M5 \
  --output-dir outputs/analysis \
  --bootstrap-samples 10000
```

The bootstrap keeps paired video multiplicities and recomputes global,
class-wise AP for every resample.

## TadTR conversion and qualitative examples

`convert_tadtr_predictions.py` converts a TadTR detection JSON to the same
schema as MEM-TAD exports. `generate_qualitative_examples.py` then compares M5,
TadTR, and Q1 predictions on video frames. Run either script with `--help` for
the required source-specific paths.

## Frame cache generation

The two cache generators are retained for experiments that start from decoded
frames rather than pre-extracted feature arrays:

```bash
python tools/generate_dataset_cache.py --help
python tools/generate_dataset_cache_lz4.py --help
```

They support inherited YAML configs through `_base_`. The main paper release
uses pre-extracted `.npy` feature files and does not require a frame cache.
