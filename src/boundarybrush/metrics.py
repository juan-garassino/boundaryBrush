"""Segmentation metrics on bool masks shaped (H, W) or (B, H, W); all per image."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from boundarybrush.ops import erode


def _batched(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    return mask[None] if mask.ndim == 2 else mask


def iou(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-image IoU; two empty masks score 1."""
    p, g = _batched(pred), _batched(gt)
    inter = (p & g).flatten(1).sum(1).float()
    union = (p | g).flatten(1).sum(1).float()
    return torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))


def boundary_width(height: int, width: int) -> int:
    """Band width d = round(2% of the image diagonal), at least 1 px (Cheng et al., 2021)."""
    return max(1, round(0.02 * (height**2 + width**2) ** 0.5))


def boundary_iou(pred: torch.Tensor, gt: torch.Tensor, d: int | None = None) -> torch.Tensor:
    """IoU restricted to each mask's inner band of width d."""
    p, g = _batched(pred), _batched(gt)
    d = d or boundary_width(p.shape[-2], p.shape[-1])
    return iou(p & ~erode(p, d), g & ~erode(g, d))


def noc(ious: Sequence[float], threshold: float = 0.85, max_clicks: int = 5) -> int:
    """Number of clicks until IoU >= threshold; failures count as max_clicks."""
    for i, value in enumerate(ious[:max_clicks], start=1):
        if value >= threshold:
            return i
    return max_clicks
