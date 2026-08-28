import torch
from tqdm import tqdm
import os
import io
import json
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import math
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.DataLoader import load_train_val_data
from models.mem_tad import mem_tad
from utils.losses import MemTADLoss
from utils.model_profile import profile_model_complexity, print_model_complexity
from utils.temporal_nms import apply_temporal_nms
from utils.config import load_config
from tqdm import tqdm
from PIL import Image


def _strip_batch_dim(x):
    if x.dim() == 3 and x.size(0) == 1:
        return x[0]
    return x


def _canonical_segments(seg):
    start = torch.minimum(seg[..., 0], seg[..., 1])
    end = torch.maximum(seg[..., 0], seg[..., 1])
    return torch.stack([start, end], dim=-1)


def _temporal_iou(pred_seg, gt_segs, eps=1e-6):
    if gt_segs.numel() == 0:
        return pred_seg.new_zeros((0,))

    pred_len = (pred_seg[1] - pred_seg[0]).clamp(min=eps)
    gt_len = (gt_segs[:, 1] - gt_segs[:, 0]).clamp(min=eps)

    inter_start = torch.maximum(pred_seg[0], gt_segs[:, 0])
    inter_end = torch.minimum(pred_seg[1], gt_segs[:, 1])
    inter = (inter_end - inter_start).clamp(min=0.0)

    union = pred_len + gt_len - inter
    return inter / (union + eps)


def _extract_pred_and_gt(
    output,
    labels,
    class_num,
    conf_threshold=0.0,
    score_mode="sigmoid_class",
    nms_enabled=False,
    nms_type="hard",
    nms_iou_threshold=0.5,
    nms_sigma=0.5,
    nms_min_score=1e-4,
    nms_max_detections=200,
    pre_nms_topk=None,
    class_selection="query_max",
):
    output = _strip_batch_dim(output).detach()
    labels = _strip_batch_dim(labels).detach()

    quality_prob = torch.sigmoid(output[:, 0])
    cls_prob = torch.sigmoid(output[:, 3:])
    pred_seg_all = _canonical_segments(output[:, 1:3])
    score_mode = str(score_mode or "sigmoid_class").strip().lower()
    class_selection = str(class_selection or "query_max").strip().lower()
    use_flatten_topk = class_selection in {"flatten_topk", "all_class_topk", "query_class_topk"}
    pre_topk = int(pre_nms_topk or nms_max_detections)
    if score_mode == "objectness":
        _, pred_cls = torch.max(cls_prob, dim=-1)
        pred_conf = quality_prob
        pred_seg = pred_seg_all
    elif score_mode in {
        "quality_x_class", "quality*class", "iou_x_class",
        "objectness_x_class", "objectness*class", "fused",
    }:
        class_scores = quality_prob[:, None] * cls_prob
        if use_flatten_topk:
            flat_scores = class_scores.flatten()
            topk = min(max(pre_topk, 1), flat_scores.numel())
            pred_conf, flat_idx = torch.topk(flat_scores, k=topk, largest=True)
            query_idx = torch.div(flat_idx, class_num, rounding_mode="floor")
            pred_cls = flat_idx % class_num
            pred_seg = pred_seg_all[query_idx]
        else:
            pred_conf, pred_cls = torch.max(class_scores, dim=-1)
            pred_seg = pred_seg_all
    else:
        if use_flatten_topk:
            flat_scores = cls_prob.flatten()
            topk = min(max(pre_topk, 1), flat_scores.numel())
            pred_conf, flat_idx = torch.topk(flat_scores, k=topk, largest=True)
            query_idx = torch.div(flat_idx, class_num, rounding_mode="floor")
            pred_cls = flat_idx % class_num
            pred_seg = pred_seg_all[query_idx]
        else:
            pred_conf, pred_cls = torch.max(cls_prob, dim=-1)
            pred_seg = pred_seg_all

    keep = pred_conf >= conf_threshold
    pred_conf = pred_conf[keep].cpu()
    pred_seg = pred_seg[keep].cpu()
    pred_cls = pred_cls[keep].cpu()

    if bool(nms_enabled):
        pred_conf, pred_seg, pred_cls = apply_temporal_nms(
            pred_conf=pred_conf,
            pred_seg=pred_seg,
            pred_cls=pred_cls,
            class_num=class_num,
            nms_type=nms_type,
            iou_threshold=float(nms_iou_threshold),
            sigma=float(nms_sigma),
            min_score=float(nms_min_score),
            max_detections=int(nms_max_detections),
        )

    if labels.numel() == 0:
        gt_seg = torch.empty((0, 2), dtype=torch.float32)
        gt_cls = torch.empty((0,), dtype=torch.long)
    else:
        gt_seg = _canonical_segments(labels[:, :2]).cpu()
        gt_cls = labels[:, 2:2 + class_num].argmax(dim=-1).cpu()

    return {
        "pred_conf": pred_conf,
        "pred_seg": pred_seg,
        "pred_cls": pred_cls,
        "gt_seg": gt_seg,
        "gt_cls": gt_cls,
    }


def _nms_kwargs_from_cfg(cfg):
    return {
        "score_mode": str(cfg.get("detection_score_mode", "sigmoid_class") or "sigmoid_class").lower(),
        "nms_enabled": bool(cfg.get("postprocess_nms_enabled", False)),
        "nms_type": str(cfg.get("postprocess_nms_type", "hard") or "hard").lower(),
        "nms_iou_threshold": float(cfg.get("postprocess_nms_iou_threshold", 0.5)),
        "nms_sigma": float(cfg.get("postprocess_nms_sigma", 0.5)),
        "nms_min_score": float(cfg.get("postprocess_nms_min_score", 1e-4)),
        "nms_max_detections": int(cfg.get("postprocess_nms_max_detections", 200)),
        "pre_nms_topk": int(cfg.get("postprocess_pre_nms_topk", cfg.get("postprocess_nms_max_detections", 200))),
        "class_selection": str(cfg.get("postprocess_class_selection", "query_max") or "query_max").lower(),
    }


def _attach_video_metadata(record, video_data):
    record["video_id"] = str(video_data["video_id"])
    record["fps"] = float(video_data.get("fps", 0.0) or 0.0)
    record["total_frames"] = int(video_data.get("total_frames", 0) or 0)
    record["duration"] = float(video_data.get("duration", 0.0) or 0.0)
    record["num_input_clips"] = int(len(video_data.get("video_clips", [])))


def _write_prediction_export(path, records, metrics, cfg, checkpoint_path):
    annotation_classes = []
    annotation_path = str(cfg.get("annotations_json_path", "") or "")
    if annotation_path and os.path.isfile(annotation_path):
        with open(annotation_path, "r", encoding="utf-8") as handle:
            annotation_classes = list(json.load(handle).get("classes", []))

    exported_records = []
    for record in records:
        exported_records.append({
            "video_id": str(record["video_id"]),
            "fps": float(record.get("fps", 0.0)),
            "total_frames": int(record.get("total_frames", 0)),
            "duration": float(record.get("duration", 0.0)),
            "num_input_clips": int(record.get("num_input_clips", 0)),
            "loss": float(record.get("loss", 0.0)),
            "predictions": [
                {
                    "segment": [float(segment[0]), float(segment[1])],
                    "label": int(label),
                    "score": float(score),
                }
                for score, segment, label in zip(
                    record["pred_conf"].tolist(),
                    record["pred_seg"].tolist(),
                    record["pred_cls"].tolist(),
                )
            ],
            "ground_truth": [
                {
                    "segment": [float(segment[0]), float(segment[1])],
                    "label": int(label),
                }
                for segment, label in zip(
                    record["gt_seg"].tolist(),
                    record["gt_cls"].tolist(),
                )
            ],
        })

    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint_path),
        "annotations_json_path": annotation_path,
        "classes": annotation_classes,
        "class_num": int(cfg["class_num"]),
        "eval_scope": str(cfg.get("map_eval_scope", "global")),
        "eval_conf_threshold": float(cfg.get("eval_conf_threshold", 0.0)),
        "postprocess": _nms_kwargs_from_cfg(cfg),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "records": exported_records,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"Exported {len(exported_records)} validation records to {path}")


def _select_state(state, indices):
    return {
        "shallow": state["shallow"].index_select(0, indices),
        "deep": None if state.get("deep", None) is None else state["deep"].index_select(0, indices),
    }


def _scatter_state(state, indices, active_state):
    new_state = {}
    for key in ("shallow", "deep"):
        value = state.get(key, None)
        active_value = active_state.get(key, None)
        if value is None or active_value is None:
            new_state[key] = value
            continue
        merged = value.clone()
        active_value = active_value.to(device=value.device, dtype=value.dtype)
        merged.index_copy_(0, indices, active_value)
        new_state[key] = merged
    return new_state


def _validate_single_prepared_video(model, criterion, video_data, cfg, prepare_clip_fn):
    labels = video_data["labels"].to(cfg["device"])
    video_clips = video_data["video_clips"]

    if len(video_clips) == 0:
        model.reset_memory()
        return 0.0, 0, [], 0

    output = None
    state = model.init_state(1, device=cfg["device"], dtype=torch.float32)
    for clip_idx, clip_tensor_cpu in enumerate(video_clips):
        clip_tensor = clip_tensor_cpu.to(cfg["device"], non_blocking=True)
        clip_tensor = prepare_clip_fn(clip_tensor, video_data)
        output, state = model(
            clip_tensor,
            state=state,
            decode=(clip_idx + 1 == len(video_clips)),
            detach_state=False,
        )

    if output is None or (not torch.isfinite(output).all()):
        model.reset_memory()
        return 0.0, 0, [], 1

    loss = criterion(output, labels)
    if not torch.isfinite(loss):
        model.reset_memory()
        return 0.0, 0, [], 1

    rec = _extract_pred_and_gt(
        output,
        labels,
        cfg["class_num"],
        conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
        **_nms_kwargs_from_cfg(cfg),
    )
    _attach_video_metadata(rec, video_data)
    rec["loss"] = float(loss.item())

    model.reset_memory()
    return float(loss.item()), 1, [rec], 0


def _validate_prepared_video_batch(model, criterion, batch_video_data, cfg, prepare_clip_fn):
    videos = batch_video_data.get("videos", None)
    if not videos:
        return _validate_single_prepared_video(model, criterion, batch_video_data, cfg, prepare_clip_fn)

    videos = [video for video in videos if len(video.get("video_clips", [])) > 0]
    if not videos:
        model.reset_memory()
        return 0.0, 0, [], 0

    batch_size = len(videos)
    max_clips = max(len(video["video_clips"]) for video in videos)
    outputs_for_loss = []
    labels_for_loss = []
    videos_for_loss = []
    reference_clip = None
    state = model.init_state(batch_size, device=cfg["device"], dtype=torch.float32)

    for step in range(max_clips):
        clip_batch = []
        decode_mask = []
        active_indices = []
        for video_idx, video in enumerate(videos):
            video_clips = video["video_clips"]
            is_active = step < len(video_clips)
            if not is_active:
                continue
            active_indices.append(video_idx)
            decode_mask.append(step + 1 == len(video_clips))
            clip_tensor = video_clips[step].to(cfg["device"], non_blocking=True)
            clip_tensor = prepare_clip_fn(clip_tensor, video)
            reference_clip = clip_tensor
            clip_batch.append(clip_tensor)

        if not clip_batch:
            continue

        clip_batch = torch.cat(clip_batch, dim=0)
        active_index_tensor = torch.tensor(active_indices, dtype=torch.long, device=cfg["device"])
        active_state = _select_state(state, active_index_tensor)
        decode_mask_tensor = torch.tensor(decode_mask, dtype=torch.bool, device=cfg["device"])

        output, active_new_state = model(
            clip_batch,
            state=active_state,
            decode_mask=decode_mask_tensor,
            detach_state=False,
        )
        state = _scatter_state(state, active_index_tensor, active_new_state)

        if output is not None and output.numel() > 0:
            ending_indices = [i for i, flag in enumerate(decode_mask) if flag]
            for local_idx, active_pos in enumerate(ending_indices):
                video_idx = active_indices[active_pos]
                labels = videos[video_idx]["labels"].to(cfg["device"])
                outputs_for_loss.append(output[local_idx])
                labels_for_loss.append(labels)
                videos_for_loss.append(videos[video_idx])

    if not outputs_for_loss:
        model.reset_memory()
        return 0.0, 0, [], 0

    records = []
    losses = []
    skipped = 0
    for output, labels, video in zip(outputs_for_loss, labels_for_loss, videos_for_loss):
        if output is None or (not torch.isfinite(output).all()):
            skipped += 1
            continue
        loss = criterion(output, labels)
        if not torch.isfinite(loss):
            skipped += 1
            continue
        losses.append(loss)
        rec = _extract_pred_and_gt(
            output,
            labels,
            cfg["class_num"],
            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
            **_nms_kwargs_from_cfg(cfg),
        )
        _attach_video_metadata(rec, video)
        rec["loss"] = float(loss.item())
        records.append(rec)

    model.reset_memory()
    if not losses:
        return 0.0, 0, records, skipped
    loss_sum = float(torch.stack(losses).sum().item())
    return loss_sum, len(losses), records, skipped


def _compute_ap(precision, recall):
    if precision.numel() == 0 or recall.numel() == 0:
        return 0.0

    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])

    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])

    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    ap = torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap.item())


def _average_map(metrics_by_iou, cfg):
    """Average the configured tIoU metrics using one shared paper protocol."""
    default_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresholds = cfg.get("avg_map_iou_thresholds", default_thresholds)
    try:
        thresholds = [round(float(value), 1) for value in thresholds]
    except (TypeError, ValueError):
        thresholds = default_thresholds
    thresholds = [value for value in thresholds if value in metrics_by_iou]
    if not thresholds:
        thresholds = default_thresholds
    return sum(metrics_by_iou[value]["mAP"] for value in thresholds) / len(thresholds)


def _compute_metrics_at_iou(video_records, class_num, iou_threshold, conf_threshold=0.0):
    class_aps = []
    overall_tp = 0
    overall_fp = 0
    overall_gt = 0

    for cls_id in range(class_num):
        gt_by_video = {}
        pred_list = []
        gt_count = 0

        for rec in video_records:
            video_id = rec["video_id"]

            gt_mask = rec["gt_cls"] == cls_id
            gt_cls_seg = rec["gt_seg"][gt_mask]
            if gt_cls_seg.numel() > 0:
                gt_by_video[video_id] = {
                    "seg": gt_cls_seg,
                    "used": torch.zeros(gt_cls_seg.size(0), dtype=torch.bool),
                }
                gt_count += gt_cls_seg.size(0)

            pred_mask = rec["pred_cls"] == cls_id
            if conf_threshold > 0:
                pred_mask = pred_mask & (rec["pred_conf"] >= float(conf_threshold))
            if pred_mask.any():
                cls_conf = rec["pred_conf"][pred_mask]
                cls_seg = rec["pred_seg"][pred_mask]
                for i in range(cls_conf.size(0)):
                    pred_list.append((float(cls_conf[i].item()), video_id, cls_seg[i]))

        if gt_count == 0:
            continue

        pred_list.sort(key=lambda x: x[0], reverse=True)
        tp = torch.zeros(len(pred_list), dtype=torch.float32)
        fp = torch.zeros(len(pred_list), dtype=torch.float32)

        for i, (_, video_id, pred_seg) in enumerate(pred_list):
            if video_id not in gt_by_video:
                fp[i] = 1.0
                continue

            gt_seg = gt_by_video[video_id]["seg"]
            used = gt_by_video[video_id]["used"]
            ious = _temporal_iou(pred_seg, gt_seg)
            if ious.numel() == 0:
                fp[i] = 1.0
                continue

            best_iou, best_idx = torch.max(ious, dim=0)
            best_idx = int(best_idx.item())

            if best_iou.item() >= iou_threshold and not used[best_idx]:
                tp[i] = 1.0
                used[best_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        precision_curve = tp_cum / (tp_cum + fp_cum + 1e-12)
        recall_curve = tp_cum / (gt_count + 1e-12)
        class_aps.append(_compute_ap(precision_curve, recall_curve))

        overall_tp += int(tp.sum().item())
        overall_fp += int(fp.sum().item())
        overall_gt += int(gt_count)

    mAP = float(sum(class_aps) / len(class_aps)) if class_aps else 0.0
    precision = float(overall_tp / (overall_tp + overall_fp + 1e-12))
    recall = float(overall_tp / (overall_gt + 1e-12))
    return precision, recall, mAP


def _compute_metrics_by_iou(
    video_records,
    class_num,
    iou_thresholds,
    conf_threshold=0.0,
    eval_scope="global",
):
    eval_scope = str(eval_scope or "global").strip().lower()
    if eval_scope not in {"global", "video"}:
        eval_scope = "global"

    metrics_by_iou = {}
    for iou_thr in iou_thresholds:
        if not video_records:
            metrics_by_iou[iou_thr] = {"precision": 0.0, "recall": 0.0, "mAP": 0.0}
            continue

        if eval_scope == "video":
            p_list = []
            r_list = []
            m_list = []
            for rec in video_records:
                p, r, m = _compute_metrics_at_iou(
                    [rec],
                    class_num,
                    iou_thr,
                    conf_threshold=conf_threshold,
                )
                p_list.append(float(p))
                r_list.append(float(r))
                m_list.append(float(m))

            metrics_by_iou[iou_thr] = {
                "precision": float(sum(p_list) / len(p_list)),
                "recall": float(sum(r_list) / len(r_list)),
                "mAP": float(sum(m_list) / len(m_list)),
            }
            continue

        p, r, m = _compute_metrics_at_iou(
            video_records,
            class_num,
            iou_thr,
            conf_threshold=conf_threshold,
        )
        metrics_by_iou[iou_thr] = {"precision": p, "recall": r, "mAP": m}

    return metrics_by_iou


def validate_one_epoch(
    model,
    dataloader_val,
    criterion,
    cfg,
    prepare_video_batch_fn,
    prepare_clip_fn=None,
    csv_path=None,
    epoch='',
    return_records=False,
):
    if prepare_clip_fn is None:
        prepare_clip_fn = lambda x, _: x

    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    video_records = []
    skipped_non_finite = 0

    if not bool(cfg.get("_validation_batch_info_printed", False)):
        print(
            ">>> Validation DataLoader: "
            f"batch_size={getattr(dataloader_val, 'batch_size', None)}, "
            f"steps={len(dataloader_val)}"
        )
        cfg["_validation_batch_info_printed"] = True

    with torch.no_grad():
        pbar_val = tqdm(total=len(dataloader_val), desc="Validation", leave=False)

        # reuse configuration flags from training loop
        async_prefetch = cfg.get("async_prefetch_next_video", False)
        if async_prefetch:
            val_iter = iter(dataloader_val)
            prefetch_queue_size = cfg.get("prefetch_videos_ahead", 2) + 1
            future_queue = deque()

            def submit_next():
                try:
                    batch = next(val_iter)
                except StopIteration:
                    return False
                future_queue.append(executor.submit(prepare_video_batch_fn, batch))
                return True

            with ThreadPoolExecutor(max_workers=cfg.get("prefetch_workers", 1)) as executor:
                # prime queue
                for _ in range(prefetch_queue_size):
                    if not submit_next():
                        break

                while future_queue:
                    video_data = future_queue.popleft().result()
                    loss_sum, count, records, skipped = _validate_prepared_video_batch(
                        model, criterion, video_data, cfg, prepare_clip_fn
                    )
                    val_loss_sum += loss_sum
                    val_count += count
                    video_records.extend(records)
                    skipped_non_finite += skipped
                    pbar_val.update(1)

                    # refill queue
                    while len(future_queue) < prefetch_queue_size:
                        if not submit_next():
                            break
        else:
            for batch_val in dataloader_val:
                video_data = prepare_video_batch_fn(batch_val)
                loss_sum, count, records, skipped = _validate_prepared_video_batch(
                    model, criterion, video_data, cfg, prepare_clip_fn
                )
                val_loss_sum += loss_sum
                val_count += count
                video_records.extend(records)
                skipped_non_finite += skipped
                pbar_val.update(1)
                
    pbar_val.close()

    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    conf_thr = float(cfg.get("eval_conf_threshold", 0.0))
    metrics_by_iou = _compute_metrics_by_iou(
        video_records,
        cfg["class_num"],
        iou_thresholds,
        conf_threshold=conf_thr,
        eval_scope=cfg.get("map_eval_scope", "global"),
    )

    if skipped_non_finite > 0:
        print(
            f"Warning: skipped {skipped_non_finite} validation videos due to non-finite output/loss."
        )

    model.train()
    avg_map = _average_map(metrics_by_iou, cfg)
    metrics = {
        "loss": val_loss_sum / max(val_count, 1),
        "precision": metrics_by_iou[0.5]["precision"],
        "recall": metrics_by_iou[0.5]["recall"],
        "map03": metrics_by_iou[0.3]["mAP"],
        "map04": metrics_by_iou[0.4]["mAP"],
        "map05": metrics_by_iou[0.5]["mAP"],
        "map06": metrics_by_iou[0.6]["mAP"],
        "map07": metrics_by_iou[0.7]["mAP"],
        "map08": metrics_by_iou[0.8]["mAP"],
        "map09": metrics_by_iou[0.9]["mAP"],
        "avg_map": avg_map,
    }
    if return_records:
        metrics["video_records"] = video_records
    return metrics
    

def validate(
    model,
    dataloader_val,
    criterion,
    cfg,
    prepare_video_batch_fn,
    prepare_clip_fn=None,
    csv_path=None
):
    if prepare_clip_fn is None:
        prepare_clip_fn = lambda x, _: x

    model.eval()
    val_loss_sum = 0.0
    val_count = 0
    video_records = []

    with torch.no_grad():
        pbar_val = tqdm(total=len(dataloader_val), desc="Validation", leave=False)

        # reuse configuration flags from training loop
        async_prefetch = cfg.get("async_prefetch_next_video", False)
        if async_prefetch:
            val_iter = iter(dataloader_val)
            prefetch_queue_size = cfg.get("prefetch_videos_ahead", 2) + 1
            future_queue = deque()

            def submit_next():
                try:
                    batch = next(val_iter)
                except StopIteration:
                    return False
                future_queue.append(executor.submit(prepare_video_batch_fn, batch))
                return True

            with ThreadPoolExecutor(max_workers=cfg.get("prefetch_workers", 1)) as executor:
                # prime queue
                for _ in range(prefetch_queue_size):
                    if not submit_next():
                        break

                while future_queue:
                    video_data = future_queue.popleft().result()

                    labels = video_data["labels"].to(cfg["device"])
                    video_clips = video_data["video_clips"]

                    if len(video_clips) == 0:
                        model.reset_memory()
                        pbar_val.update(1)
                    else:
                        output = None
                        for clip_idx, clip_tensor_cpu in enumerate(video_clips):
                            clip_tensor = clip_tensor_cpu.to(cfg["device"], non_blocking=True)
                            clip_tensor = prepare_clip_fn(clip_tensor, video_data)
                            output = model(clip_tensor, (clip_idx + 1 == len(video_clips)))

                        loss = criterion(output, labels)
                        val_loss_sum += float(loss.item())
                        val_count += 1

                        rec = _extract_pred_and_gt(
                            output,
                            labels,
                            cfg["class_num"],
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                            **_nms_kwargs_from_cfg(cfg),
                        )
                        rec["video_id"] = video_data["video_id"]
                        rec["loss"] = float(loss.item())
                        # compute per-video metrics at multiple IoUs
                        p, r, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.3,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["prec03"], rec["rec03"], rec["map03"] = p, r, m
                        _, _, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.4,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["map04"] = m
                        _, _, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.5,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["map05"] = m
                        p, r, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.6,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["map06"] = m
                        _, _, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.7,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["map07"] = m
                        _, _, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.8,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["map08"] = m
                        _, _, m = _compute_metrics_at_iou(
                            [rec], cfg["class_num"], 0.9,
                            conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                        )
                        rec["map09"] = m
                        # avg mAP
                        rec["avg_map"] = (
                            rec["map03"]
                            + rec["map04"]
                            + rec["map05"]
                            + rec["map06"]
                            + rec["map07"]
                            + rec["map08"]
                            + rec["map09"]
                        ) / 7.0

                        video_records.append(rec)

                        model.reset_memory()
                        pbar_val.update(1)

                    # refill queue
                    while len(future_queue) < prefetch_queue_size:
                        if not submit_next():
                            break
                pbar_val.close()
        else:
            for batch_val in dataloader_val:
                video_data = prepare_video_batch_fn(batch_val)
                labels = video_data["labels"].to(cfg["device"])
                video_clips = video_data["video_clips"]

                if len(video_clips) == 0:
                    model.reset_memory()
                    pbar_val.update(1)
                    continue

                output = None
                for clip_idx, clip_tensor_cpu in enumerate(video_clips):
                    clip_tensor = clip_tensor_cpu.to(cfg["device"], non_blocking=True)
                    clip_tensor = prepare_clip_fn(clip_tensor, video_data)
                    output = model(clip_tensor, (clip_idx + 1 == len(video_clips)))

                loss = criterion(output, labels)
                val_loss_sum += float(loss.item())
                val_count += 1

                rec = _extract_pred_and_gt(
                    output,
                    labels,
                    cfg["class_num"],
                    conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
                    **_nms_kwargs_from_cfg(cfg),
                )
                rec["video_id"] = video_data["video_id"]
                video_records.append(rec)

                model.reset_memory()
                pbar_val.update(1)
            pbar_val.close()

    # if csv_path specified, dump per-video records
    if csv_path is not None:
        import csv

        fieldnames = [
            "video_id", "loss", "prec03", "rec03", "map03", "map04", "map05", "map06",
            "map07", "map08", "map09", "avg_map"
        ]
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in video_records:
                row = {k: rec.get(k, "") for k in fieldnames}
                writer.writerow(row)

    iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    metrics_by_iou = _compute_metrics_by_iou(
        video_records,
        cfg["class_num"],
        iou_thresholds,
        conf_threshold=float(cfg.get("eval_conf_threshold", 0.0)),
        eval_scope=cfg.get("map_eval_scope", "global"),
    )

    model.train()
    avg_map = _average_map(metrics_by_iou, cfg)
    return {
        "loss": val_loss_sum / max(val_count, 1),
        "precision": metrics_by_iou[0.5]["precision"],
        "recall": metrics_by_iou[0.5]["recall"],
        "map03": metrics_by_iou[0.3]["mAP"],
        "map04": metrics_by_iou[0.4]["mAP"],
        "map05": metrics_by_iou[0.5]["mAP"],
        "map06": metrics_by_iou[0.6]["mAP"],
        "map07": metrics_by_iou[0.7]["mAP"],
        "map08": metrics_by_iou[0.8]["mAP"],
        "map09": metrics_by_iou[0.9]["mAP"],
        "avg_map": avg_map,
    }


def val_model(args):
    print(f"Validating model with config: {args.cfg}")
    cfg = load_config(args.cfg)

    # Reuse training-side preprocessing/config sanitation to keep eval consistent.
    from utils import trainer as trainer_utils

    cfg = trainer_utils.check_config(cfg, args.device)
    if cfg is False:
        return False
    cfg["map_eval_scope"] = str(getattr(args, "map_eval_scope", "global") or "global").strip().lower()
    if cfg["map_eval_scope"] not in {"global", "video"}:
        cfg["map_eval_scope"] = "global"

    cache_state = trainer_utils._init_dataset_cache_state(cfg)

    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"][0], cfg["input_size"][1])),
        transforms.ToTensor(),
    ])

    _, val_dataset = load_train_val_data(
        frames_dir=cfg.get("frames_dir", None),
        json_path=cfg["annotations_json_path"],
        device="cpu",
        features_dir=cfg.get("dataset_cache_dir", None),
    )
    cfg["class_num"] = val_dataset.class_num

    dataloader_val = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["dataloader_num_workers"],
        pin_memory=cfg["dataloader_pin_memory"],
        collate_fn=trainer_utils._video_collate,
    )

    model = mem_tad(cfg).to(cfg["device"])
    model_profile = profile_model_complexity(model, cfg, cfg["device"])
    print_model_complexity(model_profile, prefix="Validation")
    criterion = MemTADLoss(
        class_num=cfg["class_num"],
        conf_weight=cfg.get("loss_conf_weight", cfg.get("conf_weight", 1.0)),
        loc_weight=cfg.get("loss_loc_weight", cfg.get("loc_weight", 2.0)),
        cls_weight=cfg.get("loss_cls_weight", cfg.get("cls_weight", 1.0)),
        match_iou_threshold=cfg.get("loss_match_iou_threshold", cfg.get("match_iou_threshold", 0.1)),
        focal_gamma=cfg.get("loss_focal_gamma", 2.0),
        focal_alpha=cfg.get("loss_focal_alpha", 0.25),
        neg_pos_ratio=cfg.get("loss_neg_pos_ratio", 3.0),
        iou_loc_weight=cfg.get("loss_iou_weight", 0.5),
        label_smoothing=cfg.get("loss_label_smoothing", 0.05),
        force_gt_match=cfg.get("loss_force_gt_match", True),
        missed_gt_weight=cfg.get("loss_missed_gt_weight", 0.5),
        matcher=cfg.get("loss_matcher", "hungarian"),
        match_cost_class=cfg.get("loss_match_cost_class", 1.0),
        match_cost_l1=cfg.get("loss_match_cost_l1", 1.0),
        match_cost_iou=cfg.get("loss_match_cost_iou", 2.0),
        match_topk_per_gt=cfg.get("loss_match_topk_per_gt", 0),
        quality_focal_beta=cfg.get("loss_quality_focal_beta", 2.0),
        eps=cfg.get("loss_eps", cfg.get("eps", 1e-6)),
    )

    ckpt_path = (
        str(getattr(args, "resume", "") or "").strip()
        or str(cfg.get("resume", "") or "").strip()
        or str(cfg.get("val_checkpoint", "") or "").strip()
        or str(cfg.get("checkpoint_path", "") or "").strip()
    )
    if not ckpt_path:
        print("Error: missing checkpoint path. Pass --resume or set resume/val_checkpoint/checkpoint_path in yml.")
        return False
    if not os.path.isfile(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}")
        return False

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=cfg["device"])
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=True)

    zero_frame = trainer_utils.build_zero_frame(cfg)

    def _prepare_val_batch(batch):
        return trainer_utils.prepare_video_batch(
            batch,
            cfg,
            transform,
            zero_frame,
            cache_state,
            split_name="val",
        )

    def _prepare_clip(clip_tensor, video_data):
        if video_data.get("source") == "feature_cache":
            return clip_tensor.to(torch.float32).contiguous()
        clip_tensor = trainer_utils.decode_clip_for_model(clip_tensor, video_data, cfg)
        if video_data.get("source") != "cache":
            clip_tensor = trainer_utils._apply_imagenet_normalization(clip_tensor)
        return clip_tensor

    print(
        "Eval settings: "
        f"conf_thr={float(cfg.get('eval_conf_threshold', 0.0)):.4f}, "
        f"nms_enabled={bool(cfg.get('postprocess_nms_enabled', False))}, "
        f"nms_type={str(cfg.get('postprocess_nms_type', 'hard'))}, "
        f"nms_iou={float(cfg.get('postprocess_nms_iou_threshold', 0.5)):.4f}, "
        f"nms_sigma={float(cfg.get('postprocess_nms_sigma', 0.5)):.4f}, "
        f"nms_min_score={float(cfg.get('postprocess_nms_min_score', 1e-4)):.6f}, "
        f"nms_max_det={int(cfg.get('postprocess_nms_max_detections', 200))}, "
        f"pre_nms_topk={int(cfg.get('postprocess_pre_nms_topk', cfg.get('postprocess_nms_max_detections', 200)))}, "
        f"map_eval_scope={cfg['map_eval_scope']}, "
        f"score_mode={cfg.get('detection_score_mode', 'sigmoid_class')}, "
        f"class_selection={cfg.get('postprocess_class_selection', 'query_max')}"
    )

    predictions_out = str(getattr(args, "predictions_out", "") or "").strip()
    metrics = validate_one_epoch(
        model=model,
        dataloader_val=dataloader_val,
        criterion=criterion,
        cfg=cfg,
        prepare_video_batch_fn=_prepare_val_batch,
        prepare_clip_fn=_prepare_clip,
        csv_path=Path(cfg["output_dir"]) / "val_metrics.csv",
        epoch="test",
        return_records=bool(predictions_out),
    )

    video_records = metrics.pop("video_records", None)
    if predictions_out and video_records is not None:
        _write_prediction_export(
            predictions_out,
            video_records,
            metrics,
            cfg,
            ckpt_path,
        )

    print(
        "Validation result:\n"
        f"| Precision: {metrics.get('precision', 0.0):.4f} "
        f"| Recall: {metrics.get('recall', 0.0):.4f} "
        f"| mAP30: {metrics.get('map03', 0.0):.4f} "
        f"| mAP40: {metrics.get('map04', 0.0):.4f} "
        f"| mAP50: {metrics.get('map05', 0.0):.4f} "
        f"| mAP60: {metrics.get('map06', 0.0):.4f} "
        f"| mAP70: {metrics.get('map07', 0.0):.4f} "
        f"| mAP80: {metrics.get('map08', 0.0):.4f} "
        f"| mAP90: {metrics.get('map09', 0.0):.4f} "
        f"| Avg mAP: {metrics.get('avg_map', 0.0):.4f} |"
    )
    return metrics
