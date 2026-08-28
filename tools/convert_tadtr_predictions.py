#!/usr/bin/env python3
"""Convert TadTR ActivityNet-style detections to the MEM-TAD analysis schema."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--reference-export", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalized_segment(segment, duration):
    start = max(0.0, min(1.0, float(segment[0]) / duration))
    end = max(0.0, min(1.0, float(segment[1]) / duration))
    return [min(start, end), max(start, end)]


def main():
    args = parse_args()
    source = load_json(args.detections)
    reference = load_json(args.reference_export)
    source_annotations = load_json(args.source_annotations)

    classes = list(reference["classes"])
    label_to_id = {label: index for index, label in enumerate(classes)}
    detections = {str(key): value for key, value in source["results"].items()}
    annotation_database = source_annotations["database"]
    records = []

    for reference_record in reference["records"]:
        video_id = str(reference_record["video_id"])
        if video_id not in detections:
            raise KeyError(f"TadTR export has no predictions for video {video_id}")
        if video_id not in annotation_database:
            raise KeyError(f"TadTR annotations have no video {video_id}")

        duration = float(reference_record["duration"])
        source_duration = float(annotation_database[video_id]["duration"])
        if abs(duration - source_duration) > 1e-5:
            raise ValueError(
                f"Duration mismatch for video {video_id}: "
                f"reference={duration}, TadTR={source_duration}"
            )

        predictions = []
        for prediction in detections[video_id]:
            label = prediction["label"]
            if label not in label_to_id:
                raise ValueError(f"Unknown TadTR label {label!r} in video {video_id}")
            predictions.append(
                {
                    "segment": normalized_segment(prediction["segment"], duration),
                    "label": label_to_id[label],
                    "score": float(prediction["score"]),
                }
            )
        predictions.sort(key=lambda item: item["score"], reverse=True)

        records.append(
            {
                "video_id": video_id,
                "fps": float(reference_record["fps"]),
                "total_frames": int(reference_record["total_frames"]),
                "duration": duration,
                "num_input_clips": reference_record.get("num_input_clips"),
                "loss": None,
                "predictions": predictions,
                "ground_truth": reference_record["ground_truth"],
            }
        )

    payload = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "annotations_json_path": str(args.source_annotations),
        "classes": classes,
        "class_num": len(classes),
        "eval_scope": "external_native_predictions_on_mem_tad_val_subset",
        "eval_conf_threshold": 0.0,
        "postprocess": {
            "source_format": "ActivityNet detection JSON",
            "source_split": "TadTR historical val split (595 videos)",
            "converted_subset": "MEM-TAD val split (447 videos)",
            "native_detections_per_video": 100,
            "class_selection": "TadTR native top-k",
        },
        "metrics": {
            "native_epoch": 209,
            "native_map03": 0.1187,
            "native_map04": 0.1066,
            "native_map05": 0.0901,
            "native_map06": 0.0742,
            "native_map07": 0.0561,
            "native_avg_map": 0.0891,
        },
        "records": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(
        f"Wrote {args.output} with {len(records)} videos and "
        f"{sum(len(record['predictions']) for record in records)} predictions"
    )


if __name__ == "__main__":
    main()
