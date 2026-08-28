#!/usr/bin/env python3
"""Generate reproducible qualitative timelines from exported TennisNet predictions."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


CASE_SPECS = (
    {
        "panel": "a",
        "title": "Top-2 candidate comparison",
        "video_id": "221",
        "gt_label": 14,
        "gt_center": 0.40,
        "gt2_label": 42,
        "gt2_center": 0.435,
    },
    {
        "panel": "b",
        "title": "Boundary comparison",
        "video_id": "540",
        "gt_label": 68,
        "gt_center": 0.83,
        "gt2_label": 1,
        "gt2_center": 0.83,
    },
    {
        "panel": "c",
        "title": "Concurrent-action case",
        "video_id": "1141",
        "gt_label": 37,
        "gt_center": 0.87,
        "gt2_label": 42,
        "gt2_center": 0.70,
    },
)

COLORS = {
    "gt": "#202124",
    "other_gt": "#747b84",
    "correct": "#16856b",
    "concurrent": "#b77918",
    "wrong": "#d45b36",
    "alternative": "#3f6fb6",
}

FRAME_POSITIONS = np.linspace(0.08, 0.92, 5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m5", type=Path, required=True)
    parser.add_argument("--tadtr", type=Path, required=True)
    parser.add_argument("--q1", type=Path, required=True)
    parser.add_argument("--frames-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def temporal_iou(first, second):
    intersection = max(
        min(float(first[1]), float(second[1]))
        - max(float(first[0]), float(second[0])),
        0.0,
    )
    union = (
        float(first[1]) - float(first[0])
        + float(second[1]) - float(second[0])
        - intersection
    )
    return intersection / max(union, 1e-12)


def select_gt(record, label, target_center):
    candidates = [item for item in record["ground_truth"] if int(item["label"]) == label]
    if not candidates:
        raise ValueError(f"Video {record['video_id']} has no GT for label {label}")
    return min(
        candidates,
        key=lambda item: abs(np.mean(item["segment"]) - target_center),
    )


def top_overlapping_predictions(record, targets, count=2, min_iou=0.1):
    candidates = [
        item
        for item in record["predictions"]
        if max(
            temporal_iou(item["segment"], target["segment"])
            for target in targets
        ) >= min_iou
    ]
    return sorted(candidates, key=lambda item: float(item["score"]), reverse=True)[:count]


def best_same_class_prediction(record, target):
    candidates = [
        item
        for item in record["predictions"]
        if int(item["label"]) == int(target["label"])
    ]
    return max(
        candidates,
        key=lambda item: temporal_iou(item["segment"], target["segment"]),
        default=None,
    )


def best_matching_gt(record, prediction):
    candidates = [
        item
        for item in record["ground_truth"]
        if int(item["label"]) == int(prediction["label"])
    ]
    if not candidates:
        return None, 0.0
    match = max(
        candidates,
        key=lambda item: temporal_iou(prediction["segment"], item["segment"]),
    )
    return match, temporal_iou(prediction["segment"], match["segment"])


def match_predictions_to_targets(predictions, targets, threshold=0.5):
    pairs = []
    for prediction_index, prediction in enumerate(predictions):
        for target_index, target in enumerate(targets):
            if int(prediction["label"]) != int(target["label"]):
                continue
            iou = temporal_iou(prediction["segment"], target["segment"])
            if iou >= threshold:
                pairs.append((iou, prediction_index, target_index))
    pairs.sort(reverse=True)

    assignments = {}
    used_targets = set()
    for iou, prediction_index, target_index in pairs:
        if prediction_index in assignments or target_index in used_targets:
            continue
        assignments[prediction_index] = (target_index, iou)
        used_targets.add(target_index)
    return assignments, used_targets


def short_label(raw_label):
    if raw_label in {
        "ace", "ball_bounce", "begin", "end", "net", "net_in", "out",
        "passing_shot", "score_bottom", "score_top", "hit_bottom", "hit_top",
    }:
        return raw_label.replace("_", " ")
    fields = raw_label.split("_")
    player = f"P{fields[0]}" if fields and fields[0].isdigit() else fields[0]
    action = fields[1] if len(fields) > 1 else "action"
    hand = next((value for value in fields if value in {"forehand", "backhand"}), None)
    if action == "return" and hand:
        return f"{player} {hand} return"
    if action == "serve":
        return f"{player} serve"
    if action in {"prepare", "move", "relax"}:
        return f"{player} {action}"
    return " ".join(fields[:3])


def frame_files(frames_root, video_id):
    files = sorted((frames_root / video_id).glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No frames found for video {video_id}")
    return files


def draw_frame_strip(fig, grid, row, files, times, duration, case):
    strip_axis = fig.add_subplot(grid[row, :])
    strip_axis.set_axis_off()
    for position, time_seconds in zip(FRAME_POSITIONS, times):
        axis = strip_axis.inset_axes([position - 0.08, 0.18, 0.16, 0.78])
        normalized = np.clip(time_seconds / duration, 0.0, 1.0)
        frame_index = int(round(normalized * (len(files) - 1)))
        with Image.open(files[frame_index]) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_box_aspect(9 / 16)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_anchor("S")
        for spine in axis.spines.values():
            spine.set_visible(False)
        strip_axis.text(
            position,
            0.08,
            f"{time_seconds:.2f} s",
            transform=strip_axis.transAxes,
            fontsize=8,
            ha="center",
            va="center",
        )
    strip_position = strip_axis.get_position()
    fig.text(
        0.005,
        strip_position.y1 + 0.008,
        f"({case['panel']}) {case['title']} | Video {case['video_id']}",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def timeline_window(targets, duration):
    gt_segments = np.asarray([target["segment"] for target in targets], dtype=float)
    gt_start = float(gt_segments[:, 0].min()) * duration
    gt_end = float(gt_segments[:, 1].max()) * duration
    padding = max(0.30 * (gt_end - gt_start), 0.5)
    return max(0.0, gt_start - padding), min(duration, gt_end + padding)


def draw_segment(
    axis,
    segment,
    y,
    duration,
    color,
    label,
    *,
    dashed=False,
    height=None,
    font_size=None,
):
    start = float(segment[0]) * duration
    end = float(segment[1]) * duration
    axis.barh(
        y,
        end - start,
        left=start,
        height=height if height is not None else (0.42 if not dashed else 0.34),
        color="none" if dashed else color,
        edgecolor=color,
        linewidth=0.8,
        linestyle="--" if dashed else "-",
        zorder=3,
    )
    axis.text(
        max(start, axis.get_xlim()[0]) + 0.015 * np.diff(axis.get_xlim())[0],
        y,
        label,
        fontsize=font_size if font_size is not None else (6.8 if dashed else 7.5),
        color=color if dashed else "white",
        ha="left",
        va="center",
        clip_on=True,
        zorder=4,
    )


def prediction_label(prediction, classes, status, iou=None):
    suffix = status if iou is None else f"{status} {iou:.3f}"
    return (
        f"{short_label(classes[int(prediction['label'])])} | "
        f"{float(prediction['score']):.3f} | {suffix}"
    )


def draw_timeline(
    fig,
    grid,
    row,
    m5_record,
    tadtr_record,
    q1_record,
    targets,
    classes,
    duration,
    frame_times,
):
    axis = fig.add_subplot(grid[row, :])
    x_min, x_max = timeline_window(targets, duration)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(2.05, 8.65)
    axis.set_yticks([8.15, 7.55, 6.05, 4.50, 2.95])
    axis.set_yticklabels(["GT 1", "GT 2", "MEM-TAD", "TadTR", "Q1 FIFO"])
    axis.tick_params(axis="y", labelsize=8)
    axis.tick_params(axis="x", labelsize=8)
    axis.set_xlabel("Time (s)", fontsize=8, labelpad=1)
    axis.grid(axis="x", color="#d9dde3", linewidth=0.6, zorder=0)
    for frame_time in frame_times:
        axis.axvline(
            frame_time,
            color="#c7cdd5",
            linewidth=0.55,
            linestyle=":",
            zorder=1,
        )
    for spine_name in ("top", "right", "left"):
        axis.spines[spine_name].set_visible(False)

    draw_segment(
        axis,
        targets[0]["segment"],
        8.15,
        duration,
        COLORS["gt"],
        short_label(classes[int(targets[0]["label"])]),
    )
    draw_segment(
        axis,
        targets[1]["segment"],
        7.55,
        duration,
        COLORS["other_gt"],
        short_label(classes[int(targets[1]["label"])]),
    )

    model_rows = ((6.05, m5_record), (4.50, tadtr_record), (2.95, q1_record))
    for y, record in model_rows:
        top_predictions = top_overlapping_predictions(record, targets)
        assignments, covered_targets = match_predictions_to_targets(
            top_predictions, targets
        )
        selected_positions = (y + 0.62, y + 0.22)
        for prediction_index, top_prediction in enumerate(top_predictions):
            assignment = assignments.get(prediction_index)
            if assignment is not None:
                target_index, matched_iou = assignment
                color = COLORS["correct"]
                status = f"GT{target_index + 1}"
            else:
                _, matched_iou = best_matching_gt(m5_record, top_prediction)
                if matched_iou >= 0.5:
                    color = COLORS["concurrent"]
                    status = "other GT"
                else:
                    color = COLORS["wrong"]
                    status = "unmatched"
                    matched_iou = None
            draw_segment(
                axis,
                top_prediction["segment"],
                selected_positions[prediction_index],
                duration,
                color,
                prediction_label(
                    top_prediction,
                    classes,
                    status,
                    matched_iou,
                ),
                height=0.30,
                font_size=6.2,
            )

        missing_targets = [
            target_index
            for target_index in range(len(targets))
            if target_index not in covered_targets
        ]
        fallback_positions = (y - 0.25, y - 0.65)
        for fallback_index, target_index in enumerate(missing_targets):
            target = targets[target_index]
            alternative = best_same_class_prediction(record, target)
            if alternative is None:
                continue
            draw_segment(
                axis,
                alternative["segment"],
                fallback_positions[fallback_index],
                duration,
                COLORS["alternative"],
                (
                    f"GT{target_index + 1} "
                    f"{short_label(classes[int(target['label'])])} | "
                    f"{float(alternative['score']):.3f} | "
                    f"{temporal_iou(alternative['segment'], target['segment']):.3f}"
                ),
                dashed=True,
                height=0.374,
                font_size=6.2,
            )


def case_metadata(case, m5_record, tadtr_record, q1_record, targets, classes):
    output = {
        "panel": case["panel"],
        "case_type": case["title"],
        "video_id": case["video_id"],
        "duration": float(m5_record["duration"]),
        "ground_truths": [
            {
                "segment": target["segment"],
                "label_id": int(target["label"]),
                "label": classes[int(target["label"])],
            }
            for target in targets
        ],
    }
    for name, record in (
        ("M5", m5_record),
        ("TadTR", tadtr_record),
        ("Q1", q1_record),
    ):
        top_predictions = top_overlapping_predictions(record, targets)
        assignments, covered_targets = match_predictions_to_targets(
            top_predictions, targets
        )
        selected = []
        for prediction_index, prediction in enumerate(top_predictions):
            assignment = assignments.get(prediction_index)
            matched_gt, matched_iou = best_matching_gt(m5_record, prediction)
            selected.append(
                {
                    "prediction": prediction,
                    "displayed_gt_index": assignment[0] if assignment else None,
                    "displayed_gt_tiou": assignment[1] if assignment else None,
                    "best_all_gt_match": matched_gt,
                    "best_all_gt_tiou": matched_iou,
                }
            )
        fallback = []
        for target_index, target in enumerate(targets):
            if target_index in covered_targets:
                continue
            alternative = best_same_class_prediction(record, target)
            fallback.append(
                {
                    "gt_index": target_index,
                    "prediction": alternative,
                    "tiou": (
                        temporal_iou(alternative["segment"], target["segment"])
                        if alternative else None
                    ),
                }
            )
        output[name] = {
            "selected_top2": selected,
            "fallback_candidates": fallback,
        }
    return output


def main():
    args = parse_args()
    m5_payload = load_json(args.m5)
    tadtr_payload = load_json(args.tadtr)
    q1_payload = load_json(args.q1)
    classes = list(m5_payload["classes"])
    if classes != list(tadtr_payload["classes"]) or classes != list(q1_payload["classes"]):
        raise ValueError("M5, TadTR, and Q1 class lists do not match")
    m5_records = {str(item["video_id"]): item for item in m5_payload["records"]}
    tadtr_records = {
        str(item["video_id"]): item for item in tadtr_payload["records"]
    }
    q1_records = {str(item["video_id"]): item for item in q1_payload["records"]}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(14.0, 12.2), constrained_layout=False)
    grid = figure.add_gridspec(3, 1, hspace=0.20)
    figure.subplots_adjust(left=0.07, right=0.995, top=0.975, bottom=0.065)
    metadata = []

    for index, case in enumerate(CASE_SPECS):
        case_grid = grid[index].subgridspec(
            2,
            5,
            height_ratios=(1.45, 2.15),
            hspace=0.08,
            wspace=0.035,
        )
        m5_record = m5_records[case["video_id"]]
        tadtr_record = tadtr_records[case["video_id"]]
        q1_record = q1_records[case["video_id"]]
        targets = [
            select_gt(m5_record, case["gt_label"], case["gt_center"]),
            select_gt(m5_record, case["gt2_label"], case["gt2_center"]),
        ]
        duration = float(m5_record["duration"])
        x_min, x_max = timeline_window(targets, duration)
        frame_times = x_min + FRAME_POSITIONS * (x_max - x_min)
        files = frame_files(args.frames_root, case["video_id"])
        draw_frame_strip(figure, case_grid, 0, files, frame_times, duration, case)
        draw_timeline(
            figure,
            case_grid,
            1,
            m5_record,
            tadtr_record,
            q1_record,
            targets,
            classes,
            duration,
            frame_times,
        )
        metadata.append(
            case_metadata(
                case,
                m5_record,
                tadtr_record,
                q1_record,
                targets,
                classes,
            )
        )

    legend = (
        Patch(facecolor=COLORS["gt"], label="Ground truth 1"),
        Patch(facecolor=COLORS["other_gt"], label="Ground truth 2"),
        Patch(facecolor=COLORS["correct"], label="Top-2 matched prediction"),
        Patch(facecolor=COLORS["concurrent"], label="Top-2 other-GT prediction"),
        Patch(facecolor=COLORS["wrong"], label="Top-2 unmatched prediction"),
        Patch(
            facecolor="none",
            edgecolor=COLORS["alternative"],
            linestyle="--",
            label="Fallback GT candidate",
        ),
    )
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=6,
        frameon=False,
        fontsize=7.8,
    )
    png_path = args.output_dir / "qualitative_examples.png"
    pdf_path = args.output_dir / "qualitative_examples.pdf"
    metadata_path = args.output_dir / "qualitative_examples.json"
    figure.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
