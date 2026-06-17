import torch
import torch.nn as nn
import torch.nn.functional as F


class MemTADLoss(nn.Module):
    """Loss for temporal detection outputs.

    Prediction format:
        [max_detection_num, 3 + class_num]
        -> [confidence, start, end, class_logits...]

    Target format:
        [num_targets, 2 + class_num]
        -> [start, end, class_one_hot...]
    """

    def __init__(
        self,
        class_num,
        conf_weight=1.0,
        loc_weight=2.0,
        cls_weight=1.0,
        match_iou_threshold=0.1,
        focal_gamma=2.0,
        focal_alpha=0.25,
        neg_pos_ratio=3.0,
        iou_loc_weight=0.5,
        label_smoothing=0.05,
        eps=1e-6,
    ):
        super().__init__()
        self.class_num = self._to_int(class_num, "class_num")
        self.conf_weight = self._to_float(conf_weight, "conf_weight")
        self.loc_weight = self._to_float(loc_weight, "loc_weight")
        self.cls_weight = self._to_float(cls_weight, "cls_weight")
        self.match_iou_threshold = self._to_float(match_iou_threshold, "match_iou_threshold")
        self.focal_gamma = self._to_float(focal_gamma, "focal_gamma")
        self.focal_alpha = self._to_float(focal_alpha, "focal_alpha")
        self.neg_pos_ratio = max(self._to_float(neg_pos_ratio, "neg_pos_ratio"), 1.0)
        self.iou_loc_weight = max(self._to_float(iou_loc_weight, "iou_loc_weight"), 0.0)
        self.label_smoothing = max(min(self._to_float(label_smoothing, "label_smoothing"), 0.2), 0.0)
        self.eps = max(self._to_float(eps, "eps"), 1e-12)

    @staticmethod
    def _to_float(value, name):
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a float-like value, got {value!r}") from exc

    @staticmethod
    def _to_int(value, name):
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be an int-like value, got {value!r}") from exc

    def _strip_batch_dim(self, x):
        if x.dim() == 3 and x.size(0) == 1:
            return x[0]
        return x

    def _canonical_segments(self, seg):
        start = torch.minimum(seg[..., 0], seg[..., 1])
        end = torch.maximum(seg[..., 0], seg[..., 1])
        return torch.stack([start, end], dim=-1)

    def _temporal_iou_matrix(self, pred_seg, gt_seg):
        # pred_seg: [P, 2], gt_seg: [G, 2]
        if pred_seg.numel() == 0 or gt_seg.numel() == 0:
            return pred_seg.new_zeros((pred_seg.size(0), gt_seg.size(0)))

        pred_len = (pred_seg[:, 1] - pred_seg[:, 0]).clamp(min=self.eps)
        gt_len = (gt_seg[:, 1] - gt_seg[:, 0]).clamp(min=self.eps)

        inter_start = torch.maximum(pred_seg[:, None, 0], gt_seg[None, :, 0])
        inter_end = torch.minimum(pred_seg[:, None, 1], gt_seg[None, :, 1])
        inter = (inter_end - inter_start).clamp(min=0.0)

        union = pred_len[:, None] + gt_len[None, :] - inter
        iou = inter / (union + self.eps)
        return iou

    def _focal_conf_loss(self, pred_conf_logits, conf_target):
        pred_prob = torch.sigmoid(pred_conf_logits)
        bce = F.binary_cross_entropy_with_logits(pred_conf_logits, conf_target, reduction="none")
        pt = conf_target * pred_prob + (1.0 - conf_target) * (1.0 - pred_prob)
        focal_factor = (1.0 - pt).pow(self.focal_gamma)

        alpha_factor = conf_target.new_full(conf_target.shape, 1.0 - self.focal_alpha)
        alpha_factor = torch.where(conf_target > 0, self.focal_alpha, alpha_factor)
        weighted = alpha_factor * focal_factor * bce

        pos_mask = conf_target > 0
        neg_mask = ~pos_mask

        if pos_mask.any():
            num_pos = int(pos_mask.sum().item())
            max_neg = int(self.neg_pos_ratio * num_pos)
            neg_indices = torch.where(neg_mask)[0]

            if neg_indices.numel() > 0 and max_neg > 0:
                neg_scores = pred_prob.detach()[neg_indices]
                topk = min(max_neg, neg_indices.numel())
                hard_neg_local = torch.topk(neg_scores, k=topk, largest=True).indices
                hard_neg_idx = neg_indices[hard_neg_local]
                selected = pos_mask.clone()
                selected[hard_neg_idx] = True
                return weighted[selected].mean()

            return weighted[pos_mask].mean()

        if neg_mask.any():
            neg_indices = torch.where(neg_mask)[0]
            topk = min(32, neg_indices.numel())
            neg_scores = pred_prob.detach()[neg_indices]
            hard_neg_local = torch.topk(neg_scores, k=topk, largest=True).indices
            hard_neg_idx = neg_indices[hard_neg_local]
            return weighted[hard_neg_idx].mean()

        return weighted.mean()

    def _greedy_match(self, iou_matrix):
        # Greedy bipartite matching by IoU.
        if iou_matrix.numel() == 0:
            device = iou_matrix.device
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        p_num, g_num = iou_matrix.shape
        scores = iou_matrix.clone()
        matched_pred = []
        matched_gt = []

        while True:
            max_val, flat_idx = scores.view(-1).max(dim=0)
            if max_val.item() < self.match_iou_threshold:
                break

            pred_idx = flat_idx // g_num
            gt_idx = flat_idx % g_num
            matched_pred.append(pred_idx)
            matched_gt.append(gt_idx)

            scores[pred_idx, :] = -1.0
            scores[:, gt_idx] = -1.0

            if len(matched_pred) >= min(p_num, g_num):
                break

        if not matched_pred:
            device = iou_matrix.device
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        return torch.stack(matched_pred), torch.stack(matched_gt)

    def forward(self, pred, target):
        pred = self._strip_batch_dim(pred)
        target = self._strip_batch_dim(target)

        if pred.dim() != 2 or pred.size(-1) != self.class_num + 3:
            raise ValueError(
                f"pred shape should be [P, {self.class_num + 3}], got {tuple(pred.shape)}"
            )

        if target.numel() == 0:
            target = pred.new_zeros((0, self.class_num + 2))
        if target.dim() != 2 or target.size(-1) != self.class_num + 2:
            raise ValueError(
                f"target shape should be [G, {self.class_num + 2}], got {tuple(target.shape)}"
            )

        pred_conf_logits = pred[:, 0]
        pred_seg = self._canonical_segments(pred[:, 1:3])
        pred_cls = pred[:, 3:]

        conf_target = torch.zeros_like(pred_conf_logits)
        loc_loss = pred_conf_logits.new_zeros(())
        cls_loss = pred_conf_logits.new_zeros(())

        if target.size(0) > 0:
            gt_seg = self._canonical_segments(target[:, :2])
            gt_cls_onehot = target[:, 2:]

            iou_matrix = self._temporal_iou_matrix(pred_seg, gt_seg)
            matched_pred_idx, matched_gt_idx = self._greedy_match(iou_matrix)

            if matched_pred_idx.numel() > 0:
                # Confidence target uses matched IoU quality.
                conf_target[matched_pred_idx] = iou_matrix[
                    matched_pred_idx, matched_gt_idx
                ].detach().to(conf_target.dtype)

                loc_reg = F.smooth_l1_loss(
                    pred_seg[matched_pred_idx],
                    gt_seg[matched_gt_idx],
                    reduction="mean",
                )
                matched_iou = iou_matrix[matched_pred_idx, matched_gt_idx]
                iou_loss = 1.0 - matched_iou.mean()
                loc_loss = loc_reg + self.iou_loc_weight * iou_loss

                gt_cls_idx = gt_cls_onehot[matched_gt_idx].argmax(dim=-1)
                cls_loss = F.cross_entropy(
                    pred_cls[matched_pred_idx],
                    gt_cls_idx,
                    label_smoothing=self.label_smoothing,
                    reduction="mean",
                )

        conf_loss = self._focal_conf_loss(pred_conf_logits, conf_target)

        total_loss = (
            self.conf_weight * conf_loss
            + self.loc_weight * loc_loss
            + self.cls_weight * cls_loss
        )
        return total_loss
