import torch


def _segment_iou_1vN(seg, others, eps=1e-6):
    if others.numel() == 0:
        return seg.new_zeros((0,))

    seg_start, seg_end = seg[0], seg[1]
    seg_len = (seg_end - seg_start).clamp(min=eps)

    oth_start = others[:, 0]
    oth_end = others[:, 1]
    oth_len = (oth_end - oth_start).clamp(min=eps)

    inter_start = torch.maximum(seg_start, oth_start)
    inter_end = torch.minimum(seg_end, oth_end)
    inter = (inter_end - inter_start).clamp(min=0.0)

    union = seg_len + oth_len - inter
    return inter / (union + eps)


def _hard_nms_single_class(conf, seg, iou_threshold):
    order = torch.argsort(conf, descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        ious = _segment_iou_1vN(seg[i], seg[rest])
        rest = rest[ious <= float(iou_threshold)]
        order = rest

    if not keep:
        return conf.new_zeros((0,), dtype=torch.long)
    return torch.stack(keep)


def _soft_nms_single_class(conf, seg, sigma, min_score):
    sigma = max(float(sigma), 1e-6)
    min_score = float(min_score)

    work_conf = conf.clone()
    remaining = torch.arange(conf.size(0), device=conf.device)
    keep = []

    while remaining.numel() > 0:
        local_best = torch.argmax(work_conf[remaining])
        best_idx = remaining[local_best]
        best_score = work_conf[best_idx]

        if best_score < min_score:
            break

        keep.append(best_idx)

        # remove selected index first
        rem_mask = remaining != best_idx
        remaining = remaining[rem_mask]
        if remaining.numel() == 0:
            break

        ious = _segment_iou_1vN(seg[best_idx], seg[remaining])
        decay = torch.exp(-(ious * ious) / sigma)
        work_conf[remaining] = work_conf[remaining] * decay

    if not keep:
        return conf.new_zeros((0,), dtype=torch.long), conf.new_zeros((0,))

    keep_idx = torch.stack(keep)
    return keep_idx, work_conf[keep_idx]


def apply_temporal_nms(
    pred_conf,
    pred_seg,
    pred_cls,
    class_num,
    nms_type="hard",
    iou_threshold=0.5,
    sigma=0.5,
    min_score=1e-4,
    max_detections=200,
):
    """Apply class-wise temporal NMS and return filtered tensors.

    Inputs and outputs are CPU/GPU torch tensors with shapes:
    - pred_conf: [N]
    - pred_seg:  [N, 2]
    - pred_cls:  [N]
    """
    if pred_conf.numel() == 0:
        return pred_conf, pred_seg, pred_cls

    nms_type = str(nms_type or "hard").strip().lower()
    if nms_type not in {"hard", "soft"}:
        nms_type = "hard"

    all_conf = []
    all_seg = []
    all_cls = []

    for cls_id in range(int(class_num)):
        mask = pred_cls == cls_id
        if not mask.any():
            continue

        cls_conf = pred_conf[mask]
        cls_seg = pred_seg[mask]

        if nms_type == "hard":
            keep_idx = _hard_nms_single_class(
                conf=cls_conf,
                seg=cls_seg,
                iou_threshold=iou_threshold,
            )
            kept_conf = cls_conf[keep_idx]
            kept_seg = cls_seg[keep_idx]
        else:
            keep_idx, soft_conf = _soft_nms_single_class(
                conf=cls_conf,
                seg=cls_seg,
                sigma=sigma,
                min_score=min_score,
            )
            kept_conf = soft_conf
            kept_seg = cls_seg[keep_idx]

        if kept_conf.numel() == 0:
            continue

        keep_mask = kept_conf >= float(min_score)
        kept_conf = kept_conf[keep_mask]
        kept_seg = kept_seg[keep_mask]

        if kept_conf.numel() == 0:
            continue

        kept_cls = pred_cls.new_full((kept_conf.size(0),), int(cls_id))
        all_conf.append(kept_conf)
        all_seg.append(kept_seg)
        all_cls.append(kept_cls)

    if not all_conf:
        return (
            pred_conf.new_zeros((0,)),
            pred_seg.new_zeros((0, 2)),
            pred_cls.new_zeros((0,), dtype=pred_cls.dtype),
        )

    out_conf = torch.cat(all_conf, dim=0)
    out_seg = torch.cat(all_seg, dim=0)
    out_cls = torch.cat(all_cls, dim=0)

    # Keep global top-K by confidence after class-wise NMS.
    max_k = int(max(max_detections, 1))
    if out_conf.numel() > max_k:
        order = torch.argsort(out_conf, descending=True)[:max_k]
        out_conf = out_conf[order]
        out_seg = out_seg[order]
        out_cls = out_cls[order]

    return out_conf, out_seg, out_cls
