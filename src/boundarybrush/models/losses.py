"""Losses: pixel-weighted BCE + per-sample soft Dice, and the mini-SAM objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_bce_dice(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """logits/target/weight [N, H, W] -> per-sample loss [N]."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * weight).flatten(1).sum(1) / weight.flatten(1).sum(1).clamp(min=1e-6)
    probs = torch.sigmoid(logits)
    inter = (probs * target).flatten(1).sum(1)
    denom = probs.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = 1 - (2 * inter + 1.0) / (denom + 1.0)
    return bce + dice


def hard_iou(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """IoU of the thresholded prediction, per sample, no gradient. [N, H, W] -> [N]."""
    with torch.no_grad():
        pred = logits > 0
        gt = target > 0.5
        inter = (pred & gt).flatten(1).sum(1).float()
        union = (pred | gt).flatten(1).sum(1).float()
        return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def minisam_loss(
    masks: torch.Tensor, iou_pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    """masks [N, M, H, W], iou_pred [N, M], target/weight [N, H, W].

    Mask loss backpropagates only through the best of the M heads (SAM's min-loss); the IoU
    head is supervised on *every* head so its argmax is meaningful at inference.
    """
    n, m = masks.shape[:2]
    flat_t = target[:, None].expand(-1, m, -1, -1).flatten(0, 1)
    flat_w = weight[:, None].expand(-1, m, -1, -1).flatten(0, 1)
    per_head = weighted_bce_dice(masks.flatten(0, 1), flat_t, flat_w).view(n, m)
    best, _ = per_head.min(dim=1)
    actual = hard_iou(masks.flatten(0, 1), flat_t).view(n, m)
    iou_loss = F.mse_loss(iou_pred, actual)
    loss = best.mean() + iou_loss
    return loss, {
        "mask_loss": float(best.mean()),
        "iou_loss": float(iou_loss),
        "iou": float(actual.max(1)[0].mean()),
    }
