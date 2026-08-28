#!/usr/bin/env python3
"""Analyze exported TennisNet detections and run paired video bootstrap."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


IOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
EVENT_CLASSES = {
    "ace", "ball_bounce", "begin", "end", "hit_bottom", "hit_top",
    "net", "net_in", "out", "passing_shot", "score_bottom", "score_top",
}
ERROR_ORDER = (
    "correct", "duplicate", "classification", "localization",
    "class_and_localization", "background",
)


def parse_named_paths(values):
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        parsed[name.strip()] = Path(path).expanduser().resolve()
    return parsed


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prediction_export(path):
    payload = load_json(path)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported prediction schema in {path}")
    payload["records"] = sorted(payload["records"], key=lambda item: str(item["video_id"]))
    return payload


def segment_iou(segment, segments):
    if len(segments) == 0:
        return np.empty(0, dtype=np.float64)
    segment = np.asarray(segment, dtype=np.float64)
    segments = np.asarray(segments, dtype=np.float64)
    inter = np.maximum(
        np.minimum(segment[1], segments[:, 1]) - np.maximum(segment[0], segments[:, 0]),
        0.0,
    )
    segment_len = max(segment[1] - segment[0], 1e-12)
    segment_lens = np.maximum(segments[:, 1] - segments[:, 0], 1e-12)
    return inter / np.maximum(segment_len + segment_lens - inter, 1e-12)


def interpolated_ap(tp, fp, gt_count):
    if gt_count <= 0 or len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp, dtype=np.float64)
    fp_cum = np.cumsum(fp, dtype=np.float64)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    recall = tp_cum / float(gt_count)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def class_rank_data(records, class_id, threshold, video_to_index=None):
    gt_by_video = {}
    gt_counts = None
    if video_to_index is not None:
        gt_counts = np.zeros(len(video_to_index), dtype=np.int32)
    predictions = []

    for record in records:
        video_id = str(record["video_id"])
        gt_segments = np.asarray(
            [item["segment"] for item in record["ground_truth"] if int(item["label"]) == class_id],
            dtype=np.float64,
        ).reshape(-1, 2)
        if len(gt_segments):
            gt_by_video[video_id] = {
                "segments": gt_segments,
                "used": np.zeros(len(gt_segments), dtype=bool),
            }
            if gt_counts is not None:
                gt_counts[video_to_index[video_id]] = len(gt_segments)

        for prediction in record["predictions"]:
            if int(prediction["label"]) == class_id:
                predictions.append((
                    float(prediction["score"]),
                    video_id,
                    np.asarray(prediction["segment"], dtype=np.float64),
                ))

    predictions.sort(key=lambda item: item[0], reverse=True)
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)
    pred_video = np.empty(len(predictions), dtype=np.int32) if video_to_index is not None else None

    for index, (_, video_id, pred_segment) in enumerate(predictions):
        if pred_video is not None:
            pred_video[index] = video_to_index[video_id]
        target = gt_by_video.get(video_id)
        if target is None:
            fp[index] = 1.0
            continue
        ious = segment_iou(pred_segment, target["segments"])
        best_index = int(np.argmax(ious))
        if ious[best_index] >= threshold and not target["used"][best_index]:
            tp[index] = 1.0
            target["used"][best_index] = True
        else:
            fp[index] = 1.0

    gt_count = int(sum(len(item["segments"]) for item in gt_by_video.values()))
    return {
        "tp": tp,
        "fp": fp,
        "gt_count": gt_count,
        "gt_counts": gt_counts,
        "pred_video": pred_video,
    }


def evaluate_global(payload, thresholds=IOU_THRESHOLDS):
    records = payload["records"]
    class_num = int(payload["class_num"])
    class_ap = np.full((len(thresholds), class_num), np.nan, dtype=np.float64)
    maps = []
    for threshold_index, threshold in enumerate(thresholds):
        for class_id in range(class_num):
            rank = class_rank_data(records, class_id, threshold)
            if rank["gt_count"] == 0:
                continue
            class_ap[threshold_index, class_id] = interpolated_ap(
                rank["tp"], rank["fp"], rank["gt_count"]
            )
        maps.append(float(np.nanmean(class_ap[threshold_index])))
    return {
        "thresholds": list(thresholds),
        "maps": maps,
        "avg_map": float(np.mean(maps)),
        "class_ap": class_ap,
    }


def verify_middle_exports(payloads):
    reference_name = next(iter(payloads))
    reference = payloads[reference_name]
    reference_ids = [str(record["video_id"]) for record in reference["records"]]
    reference_gt = {
        str(record["video_id"]): record["ground_truth"] for record in reference["records"]
    }
    for name, payload in payloads.items():
        ids = [str(record["video_id"]) for record in payload["records"]]
        if ids != reference_ids:
            raise ValueError(f"Video set/order mismatch: {reference_name} vs {name}")
        if list(payload.get("classes", [])) != list(reference.get("classes", [])):
            raise ValueError(f"Class list mismatch: {reference_name} vs {name}")
        for record in payload["records"]:
            if record["ground_truth"] != reference_gt[str(record["video_id"])]:
                raise ValueError(f"GT mismatch in video {record['video_id']}: {reference_name} vs {name}")


def training_statistics(annotation_path):
    annotations = load_json(annotation_path)
    durations = []
    class_counts = Counter()
    for video in annotations["database"].values():
        if video["subset"] != "train":
            continue
        video_duration = float(video["duration"])
        for item in video["annotations"]:
            segment = item["segment"]
            if item["label"] not in EVENT_CLASSES:
                durations.append(max(float(segment[1]) - float(segment[0]), 0.0) * video_duration)
            class_counts[item["label"]] += 1
    q33, q67 = np.quantile(np.asarray(durations, dtype=np.float64), [0.33, 0.67])
    return float(q33), float(q67), class_counts


def one_to_one_gt_matches(record, threshold=0.5):
    ground_truth = record["ground_truth"]
    matched = np.zeros(len(ground_truth), dtype=bool)
    predictions = sorted(record["predictions"], key=lambda item: float(item["score"]), reverse=True)
    for prediction in predictions:
        candidates = [
            index for index, target in enumerate(ground_truth)
            if int(target["label"]) == int(prediction["label"])
        ]
        if not candidates:
            continue
        ious = segment_iou(
            prediction["segment"],
            [ground_truth[index]["segment"] for index in candidates],
        )
        best_local = int(np.argmax(ious))
        best_index = candidates[best_local]
        if ious[best_local] >= threshold and not matched[best_index]:
            matched[best_index] = True
    return matched


def duration_bin(duration, q33, q67):
    if duration <= q33:
        return "short"
    if duration <= q67:
        return "medium"
    return "long"


def duration_analysis(model_name, payload, q33, q67):
    values = defaultdict(list)
    classes = list(payload["classes"])
    for record in payload["records"]:
        gt_matched = one_to_one_gt_matches(record)
        video_duration = float(record.get("duration", 0.0))
        predictions = record["predictions"]
        num_clips = int(record.get("num_input_clips", 0))
        for gt_index, target in enumerate(record["ground_truth"]):
            if classes[int(target["label"])] in EVENT_CLASSES:
                continue
            segment = np.asarray(target["segment"], dtype=np.float64)
            gt_duration = max(segment[1] - segment[0], 0.0) * video_duration
            bin_name = duration_bin(gt_duration, q33, q67)
            same_class = [
                prediction for prediction in predictions
                if int(prediction["label"]) == int(target["label"])
            ]
            if same_class:
                ious = segment_iou(segment, [item["segment"] for item in same_class])
                best_prediction = same_class[int(np.argmax(ious))]
                best_iou = float(np.max(ious))
                start_error = abs(float(best_prediction["segment"][0]) - segment[0]) * video_duration
                end_error = abs(float(best_prediction["segment"][1]) - segment[1]) * video_duration
                has_same_prediction = 1.0
            else:
                best_iou = 0.0
                start_error = np.nan
                end_error = np.nan
                has_same_prediction = 0.0
            crosses_boundary = False
            if num_clips > 1:
                boundaries = np.arange(1, num_clips, dtype=np.float64) / num_clips
                crosses_boundary = bool(np.any((boundaries > segment[0]) & (boundaries < segment[1])))
            item = {
                "matched": float(gt_matched[gt_index]),
                "best_iou": best_iou,
                "start_error": start_error,
                "end_error": end_error,
                "has_same_prediction": has_same_prediction,
            }
            values[("duration", bin_name)].append(item)
            values[("boundary", "crossing" if crosses_boundary else "contained")].append(item)

    rows = []
    for (group_type, group), items in values.items():
        starts = np.asarray([item["start_error"] for item in items], dtype=np.float64)
        ends = np.asarray([item["end_error"] for item in items], dtype=np.float64)
        rows.append({
            "model": model_name,
            "group_type": group_type,
            "group": group,
            "num_gt": len(items),
            "recall_at_05": float(np.mean([item["matched"] for item in items])),
            "mean_best_same_class_iou": float(np.mean([item["best_iou"] for item in items])),
            "same_class_prediction_coverage": float(np.mean([item["has_same_prediction"] for item in items])),
            "start_mae_seconds": float(np.nanmean(starts)) if np.any(np.isfinite(starts)) else np.nan,
            "end_mae_seconds": float(np.nanmean(ends)) if np.any(np.isfinite(ends)) else np.nan,
        })
    order = {"short": 0, "medium": 1, "long": 2, "contained": 3, "crossing": 4}
    return sorted(rows, key=lambda row: (row["group_type"], order[row["group"]]))


def prediction_error_counts(payload, threshold=0.5, background_iou=0.1):
    counts = Counter()
    for record in payload["records"]:
        targets = record["ground_truth"]
        gt_segments = np.asarray([item["segment"] for item in targets], dtype=np.float64).reshape(-1, 2)
        used = np.zeros(len(targets), dtype=bool)
        predictions = sorted(record["predictions"], key=lambda item: float(item["score"]), reverse=True)
        for prediction in predictions:
            if not targets:
                counts["background"] += 1
                continue
            ious = segment_iou(prediction["segment"], gt_segments)
            same_indices = np.asarray([
                index for index, target in enumerate(targets)
                if int(target["label"]) == int(prediction["label"])
            ], dtype=np.int32)
            if len(same_indices):
                same_ious = ious[same_indices]
                same_local = int(np.argmax(same_ious))
                same_index = int(same_indices[same_local])
                best_same_iou = float(same_ious[same_local])
            else:
                same_index = -1
                best_same_iou = -1.0

            if best_same_iou >= threshold:
                if used[same_index]:
                    counts["duplicate"] += 1
                else:
                    counts["correct"] += 1
                    used[same_index] = True
                continue

            best_any_index = int(np.argmax(ious))
            best_any_iou = float(ious[best_any_index])
            best_any_same = int(targets[best_any_index]["label"]) == int(prediction["label"])
            if best_any_iou >= threshold and not best_any_same:
                counts["classification"] += 1
            elif best_same_iou >= background_iou:
                counts["localization"] += 1
            elif best_any_iou >= background_iou:
                counts["class_and_localization"] += 1
            else:
                counts["background"] += 1
    return counts


def frequency_groups(classes, class_counts):
    player_classes = [name for name in classes if name not in EVENT_CLASSES]
    ranked = sorted(player_classes, key=lambda name: (class_counts.get(name, 0), name), reverse=True)
    groups = {}
    splits = np.array_split(np.asarray(ranked, dtype=object), 3)
    for group_name, names in zip(("head", "middle", "tail"), splits):
        for name in names.tolist():
            groups[str(name)] = group_name
    for name in classes:
        if name in EVENT_CLASSES:
            groups[name] = "event"
    return groups


def category_analysis(payloads, evaluations, class_counts):
    classes = list(next(iter(payloads.values()))["classes"])
    groups = frequency_groups(classes, class_counts)
    detail_rows = []
    group_rows = []
    for model_name, evaluation in evaluations.items():
        per_class = np.full(evaluation["class_ap"].shape[1], np.nan, dtype=np.float64)
        valid_classes = np.any(np.isfinite(evaluation["class_ap"]), axis=0)
        per_class[valid_classes] = np.nanmean(
            evaluation["class_ap"][:, valid_classes], axis=0
        )
        grouped_values = defaultdict(list)
        for class_id, class_name in enumerate(classes):
            value = float(per_class[class_id])
            if np.isfinite(value):
                grouped_values[groups[class_name]].append(
                    (value, int(class_counts.get(class_name, 0)))
                )
            detail_rows.append({
                "model": model_name,
                "class_id": class_id,
                "class_name": class_name,
                "category_group": groups[class_name],
                "train_instances": int(class_counts.get(class_name, 0)),
                "avg_ap_03_07": value,
            })
        for group_name in ("event", "head", "middle", "tail"):
            values = [item[0] for item in grouped_values[group_name]]
            weights = np.asarray([item[1] for item in grouped_values[group_name]], dtype=np.float64)
            group_rows.append({
                "model": model_name,
                "category_group": group_name,
                "num_classes": len(values),
                "mean_class_ap_03_07": float(np.mean(values)),
                "median_class_ap_03_07": float(np.median(values)),
                "train_frequency_weighted_ap_03_07": float(
                    np.average(values, weights=weights)
                ) if np.sum(weights) > 0 else np.nan,
            })
    return detail_rows, group_rows


def rank_concentration_analysis(model_name, payload, threshold=0.5):
    fractions = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
    class_ranks = [
        class_rank_data(payload["records"], class_id, threshold)
        for class_id in range(int(payload["class_num"]))
    ]
    class_ranks = [rank for rank in class_ranks if rank["gt_count"] > 0]
    total_detected_tp = sum(float(np.sum(rank["tp"])) for rank in class_ranks)
    rows = []
    for fraction in fractions:
        tp = 0.0
        fp = 0.0
        gt = 0
        retained = 0
        for rank in class_ranks:
            keep = int(np.ceil(len(rank["tp"]) * fraction))
            tp += float(np.sum(rank["tp"][:keep]))
            fp += float(np.sum(rank["fp"][:keep]))
            gt += int(rank["gt_count"])
            retained += keep
        rows.append({
            "model": model_name,
            "per_class_rank_fraction": fraction,
            "retained_predictions": retained,
            "precision_at_05": tp / max(tp + fp, 1.0),
            "recall_at_05": tp / max(gt, 1),
            "share_of_all_detected_tp": tp / max(total_detected_tp, 1.0),
        })
    return rows


def official_prefix_mapping(source_classes, target_classes):
    mapped_names = []
    for source_name in source_classes:
        if source_name in EVENT_CLASSES:
            mapped_names.append(source_name)
            continue
        candidates = [
            target_name for target_name in target_classes
            if target_name not in EVENT_CLASSES
            and (source_name == target_name or source_name.startswith(target_name + "_"))
        ]
        if not candidates:
            raise ValueError(f"Cannot map hierarchical label {source_name}")
        mapped_names.append(max(candidates, key=len))
    return mapped_names


def remap_payload(payload, mapped_names, classes=None):
    source_classes = list(payload["classes"])
    if len(mapped_names) != len(source_classes):
        raise ValueError("Mapped label count does not match source classes")
    classes = list(classes or dict.fromkeys(mapped_names))
    class_to_id = {name: index for index, name in enumerate(classes)}
    label_map = {index: class_to_id[name] for index, name in enumerate(mapped_names)}
    records = []
    for record in payload["records"]:
        records.append({
            **{key: value for key, value in record.items() if key not in {"predictions", "ground_truth"}},
            "predictions": [
                {**item, "label": label_map[int(item["label"])]}
                for item in record["predictions"]
            ],
            "ground_truth": [
                {**item, "label": label_map[int(item["label"])]}
                for item in record["ground_truth"]
            ],
        })
    return {**payload, "classes": classes, "class_num": len(classes), "records": records}


def granularity_analysis(payloads):
    rows = []
    for granularity, payload in payloads.items():
        native = evaluate_global(payload)
        errors = prediction_error_counts(payload)
        total_predictions = sum(errors.values())
        rows.append({
            "source_granularity": granularity,
            "evaluation_level": "native",
            "num_classes": int(payload["class_num"]),
            "avg_map_03_07": native["avg_map"],
            "correct_prediction_rate": errors["correct"] / max(total_predictions, 1),
            "classification_error_rate": errors["classification"] / max(total_predictions, 1),
            "localization_error_rate": errors["localization"] / max(total_predictions, 1),
        })

    high = payloads.get("high")
    middle = payloads.get("middle")
    low = payloads.get("low")
    if high is not None and middle is not None and low is not None:
        high_classes = list(high["classes"])
        middle_classes = list(middle["classes"])
        low_classes = list(low["classes"])
        high_to_middle = official_prefix_mapping(high_classes, middle_classes)
        high_to_low = official_prefix_mapping(high_classes, low_classes)
        stroke_names = []
        for middle_name, low_name in zip(high_to_middle, high_to_low):
            if middle_name in EVENT_CLASSES:
                stroke_names.append(middle_name)
                continue
            remainder = middle_name[len(low_name):].lstrip("_")
            stroke = remainder.split("_", 1)[0] if remainder else "none"
            stroke_names.append(f"{low_name}_{stroke}")

        levels = (
            ("base", high_to_low, low_classes),
            ("stroke", stroke_names, None),
            ("technique", high_to_middle, middle_classes),
            ("spin", high_classes, high_classes),
        )
        for level, mapped_names, target_classes in levels:
            collapsed = remap_payload(high, mapped_names, target_classes)
            evaluation = evaluate_global(collapsed)
            rows.append({
                "source_granularity": "high",
                "evaluation_level": level,
                "num_classes": int(collapsed["class_num"]),
                "avg_map_03_07": evaluation["avg_map"],
                "correct_prediction_rate": np.nan,
                "classification_error_rate": np.nan,
                "localization_error_rate": np.nan,
            })
    return rows


def prepare_bootstrap_rankings(payload):
    records = payload["records"]
    video_ids = [str(record["video_id"]) for record in records]
    video_to_index = {video_id: index for index, video_id in enumerate(video_ids)}
    rankings = []
    for threshold in IOU_THRESHOLDS:
        threshold_rankings = []
        for class_id in range(int(payload["class_num"])):
            rank = class_rank_data(records, class_id, threshold, video_to_index=video_to_index)
            if rank["gt_count"] > 0:
                threshold_rankings.append(rank)
        rankings.append(threshold_rankings)
    return rankings


def bootstrap_global_map(payload, sample_counts, batch_size=512):
    rankings = prepare_bootstrap_rankings(payload)
    num_samples = sample_counts.shape[0]
    threshold_scores = np.zeros((num_samples, len(IOU_THRESHOLDS)), dtype=np.float32)

    for threshold_index, class_rankings in enumerate(rankings):
        class_sum = np.zeros(num_samples, dtype=np.float64)
        class_count = np.zeros(num_samples, dtype=np.int32)
        for rank in class_rankings:
            gt_counts = rank["gt_counts"].astype(np.int64, copy=False)
            pred_video = rank["pred_video"]
            tp_mask = rank["tp"].astype(np.float32, copy=False)
            for start in range(0, num_samples, batch_size):
                stop = min(start + batch_size, num_samples)
                weights = sample_counts[start:stop]
                total_gt = weights @ gt_counts
                valid = total_gt > 0
                class_count[start:stop] += valid.astype(np.int32)
                if len(pred_video) == 0:
                    continue
                pred_weights = weights[:, pred_video].astype(np.float32, copy=False)
                tp_increment = pred_weights * tp_mask[None, :]
                fp_increment = pred_weights - tp_increment
                tp_cum = np.cumsum(tp_increment, axis=1, dtype=np.float32)
                fp_cum = np.cumsum(fp_increment, axis=1, dtype=np.float32)
                precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
                precision = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
                denominator = np.maximum(total_gt.astype(np.float32), 1.0)[:, None]
                ap = np.sum((tp_increment / denominator) * precision, axis=1, dtype=np.float64)
                class_sum[start:stop] += np.where(valid, ap, 0.0)
        threshold_scores[:, threshold_index] = (
            class_sum / np.maximum(class_count, 1)
        ).astype(np.float32)
    return threshold_scores


def percentile_interval(values):
    lower, upper = np.quantile(values, [0.025, 0.975])
    return float(lower), float(upper)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir, duration_rows, error_rows, model_order):
    try:
        import matplotlib
    except ImportError:
        print("matplotlib is not installed; skipping analysis plots", flush=True)
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    duration_groups = ("short", "medium", "long")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(duration_groups))
    width = 0.8 / len(model_order)
    for model_index, model in enumerate(model_order):
        rows = {
            row["group"]: row for row in duration_rows
            if row["model"] == model and row["group_type"] == "duration"
        }
        offset = (model_index - (len(model_order) - 1) / 2) * width
        axes[0].bar(
            x + offset,
            [100.0 * rows[group]["recall_at_05"] for group in duration_groups],
            width=width,
            label=model,
        )
        axes[1].bar(
            x + offset,
            [rows[group]["mean_best_same_class_iou"] for group in duration_groups],
            width=width,
            label=model,
        )
    axes[0].set_ylabel("Recall@0.5 (%)")
    axes[1].set_ylabel("Mean best same-class tIoU")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels([group.capitalize() for group in duration_groups])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "duration_error_analysis.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "duration_error_analysis.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.8, 3.5))
    x = np.arange(len(model_order))
    bottom = np.zeros(len(model_order), dtype=np.float64)
    for error_type in ERROR_ORDER:
        values = np.asarray([
            next(row["fraction"] for row in error_rows if row["model"] == model and row["error_type"] == error_type)
            for model in model_order
        ])
        axis.bar(x, 100.0 * values, bottom=100.0 * bottom, label=error_type.replace("_", " "))
        bottom += values
    axis.set_xticks(x)
    axis.set_xticklabels(model_order)
    axis.set_ylabel("Share of retained predictions (%)")
    axis.legend(frameon=False, ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_error_composition.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "prediction_error_composition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--predictions", nargs="+", required=True, metavar="NAME=PATH")
    parser.add_argument("--main-model", default="M5")
    parser.add_argument("--granularity-predictions", nargs="*", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=3407)
    parser.add_argument("--bootstrap-batch-size", type=int, default=512)
    args = parser.parse_args()

    prediction_paths = parse_named_paths(args.predictions)
    payloads = {name: load_prediction_export(path) for name, path in prediction_paths.items()}
    if args.main_model not in payloads:
        raise ValueError(f"Main model {args.main_model} is not in --predictions")
    verify_middle_exports(payloads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q33, q67, class_counts = training_statistics(args.annotations)
    evaluations = {name: evaluate_global(payload) for name, payload in payloads.items()}
    summary_rows = []
    for name, evaluation in evaluations.items():
        summary_rows.append({
            "model": name,
            **{f"map_{int(threshold * 100)}": value for threshold, value in zip(IOU_THRESHOLDS, evaluation["maps"])},
            "avg_map_03_07": evaluation["avg_map"],
        })
    write_csv(args.output_dir / "global_metrics.csv", summary_rows)

    duration_rows = []
    error_rows = []
    rank_rows = []
    for name, payload in payloads.items():
        duration_rows.extend(duration_analysis(name, payload, q33, q67))
        rank_rows.extend(rank_concentration_analysis(name, payload))
        counts = prediction_error_counts(payload)
        total = sum(counts.values())
        for error_type in ERROR_ORDER:
            error_rows.append({
                "model": name,
                "error_type": error_type,
                "count": int(counts[error_type]),
                "fraction": float(counts[error_type] / max(total, 1)),
            })
    write_csv(args.output_dir / "duration_and_boundary_analysis.csv", duration_rows)
    write_csv(args.output_dir / "prediction_error_types.csv", error_rows)
    write_csv(args.output_dir / "rank_concentration.csv", rank_rows)

    class_rows, group_rows = category_analysis(payloads, evaluations, class_counts)
    write_csv(args.output_dir / "per_class_ap.csv", class_rows)
    write_csv(args.output_dir / "category_group_ap.csv", group_rows)

    main_class = {
        row["class_name"]: row["avg_ap_03_07"]
        for row in class_rows if row["model"] == args.main_model
    }
    gain_rows = []
    for baseline in payloads:
        if baseline == args.main_model:
            continue
        baseline_class = {
            row["class_name"]: row["avg_ap_03_07"]
            for row in class_rows if row["model"] == baseline
        }
        for class_name, value in main_class.items():
            gain_rows.append({
                "baseline": baseline,
                "class_name": class_name,
                "train_instances": int(class_counts.get(class_name, 0)),
                "main_ap": value,
                "baseline_ap": baseline_class[class_name],
                "delta_ap": value - baseline_class[class_name],
            })
    gain_rows.sort(key=lambda row: (row["baseline"], -abs(row["delta_ap"])))
    write_csv(args.output_dir / "per_class_gains.csv", gain_rows)

    granularity_paths = parse_named_paths(args.granularity_predictions)
    granularity_payloads = {
        name.lower(): load_prediction_export(path) for name, path in granularity_paths.items()
    }
    if granularity_payloads:
        granularity_rows = granularity_analysis(granularity_payloads)
        write_csv(args.output_dir / "granularity_analysis.csv", granularity_rows)

    video_count = len(next(iter(payloads.values()))["records"])
    rng = np.random.default_rng(args.bootstrap_seed)
    probabilities = np.full(video_count, 1.0 / video_count, dtype=np.float64)
    sample_counts = rng.multinomial(
        video_count,
        probabilities,
        size=args.bootstrap_samples,
    ).astype(np.int16)
    bootstrap_scores = {}
    for name, payload in payloads.items():
        print(f"Bootstrap global AP: {name}", flush=True)
        bootstrap_scores[name] = bootstrap_global_map(
            payload,
            sample_counts,
            batch_size=args.bootstrap_batch_size,
        )

    bootstrap_rows = []
    metric_names = [f"mAP@{threshold:.1f}" for threshold in IOU_THRESHOLDS] + ["Avg.mAP@0.3:0.7"]
    point_values = {
        name: np.asarray(evaluation["maps"] + [evaluation["avg_map"]], dtype=np.float64)
        for name, evaluation in evaluations.items()
    }
    sample_values = {
        name: np.concatenate((scores, np.mean(scores, axis=1, keepdims=True)), axis=1)
        for name, scores in bootstrap_scores.items()
    }
    for baseline in payloads:
        if baseline == args.main_model:
            continue
        differences = sample_values[args.main_model] - sample_values[baseline]
        point_difference = point_values[args.main_model] - point_values[baseline]
        for metric_index, metric_name in enumerate(metric_names):
            lower, upper = percentile_interval(differences[:, metric_index])
            bootstrap_rows.append({
                "main_model": args.main_model,
                "baseline": baseline,
                "metric": metric_name,
                "point_delta": float(point_difference[metric_index]),
                "bootstrap_mean_delta": float(np.mean(differences[:, metric_index])),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "probability_delta_gt_zero": float(np.mean(differences[:, metric_index] > 0.0)),
                "num_resamples": int(args.bootstrap_samples),
            })
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    np.savez_compressed(
        args.output_dir / "bootstrap_samples.npz",
        **{name: values for name, values in sample_values.items()},
    )

    model_order = list(payloads)
    make_plots(args.output_dir, duration_rows, error_rows, model_order)
    metadata = {
        "duration_thresholds_seconds": {"short_medium": q33, "medium_long": q67},
        "iou_thresholds": list(IOU_THRESHOLDS),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "models": {name: str(path) for name, path in prediction_paths.items()},
        "granularity_models": {name: str(path) for name, path in granularity_paths.items()},
    }
    with open(args.output_dir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
