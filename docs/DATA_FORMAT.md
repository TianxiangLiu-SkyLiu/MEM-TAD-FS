# Data format

## Annotation JSON

Each granularity uses one ActivityNet-style JSON file:

```json
{
  "classes": ["class_a", "class_b"],
  "database": {
    "video_0001": {
      "subset": "train",
      "fps": 25.0,
      "duration": 120.0,
      "frame_num": 3000,
      "annotations": [
        {
          "segment": [0.125, 0.175],
          "label": "class_a"
        }
      ]
    }
  }
}
```

Required fields:

- `classes`: ordered class-name list shared by train and validation videos;
- `subset`: `train` or `val`;
- `fps`: source-video frame rate;
- `duration`: video duration in seconds;
- `frame_num`: total source-video frames;
- `annotations`: temporal segments and labels.

Segments are normalized to `[0,1]`. Overlapping ground-truth segments are
valid, including multiple labels at the same time. A label must occur in the
top-level `classes` list.

## Feature files

Each annotation key must have a matching `<video_id>.npy` or `<video_id>.pt`
file. Supported arrays are:

- `[N,C,T,H,W]`;
- `[N,C,H,W]`, converted to `[N,C,1,H,W]`;
- `[N,H,W]`, converted to `[N,1,1,H,W]`.

`N` is the number of chronological feature blocks. Files are loaded in that
order and recurrent memory is propagated between consecutive blocks. The
configured `feature_size` is `[C,T,H,W]`; a mismatch raises an error before
training.

Paper configurations use the following layouts:

| Feature | `[C,T,H,W]` |
| --- | --- |
| SlowFast | `[2304,1,17,34]` |
| I3D | `[512,1,28,56]` |
| C3D | `[512,1,28,28]` |
| TSP | `[48,1,28,28]` |

The feature extractor and temporal sampling convention must remain consistent
across training and validation.
