"""Interactive-segmentation evaluation, written once against the Segmenter interface.

Protocol (RITM-style, deterministic): click 1 is positive at the deepest point of the ground
truth; each next click goes to the deepest point of the larger error region — positive if it
is missed foreground (FN), negative if it is spurious foreground (FP). Also IoU with the tight
ground-truth box, and (U-Net only) automatic IoU with no prompt.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import torch

from boundarybrush import metrics
from boundarybrush.domain.models import NEGATIVE, POSITIVE, Prompt
from boundarybrush.ops import deepest_point, depth_map
from boundarybrush.providers.base import Segmenter

logger = logging.getLogger(__name__)


def next_click(pred: torch.Tensor, gt: torch.Tensor) -> tuple[tuple[float, float], int] | None:
    """Deepest point of the larger error region; None when prediction == ground truth."""
    fn, fp = gt & ~pred, pred & ~gt
    if not fn.any() and not fp.any():
        return None
    d_fn = int(depth_map(fn).max()) if fn.any() else 0
    d_fp = int(depth_map(fp).max()) if fp.any() else 0
    if d_fn >= d_fp:
        return deepest_point(fn), POSITIVE
    return deepest_point(fp), NEGATIVE


def tight_box(gt: torch.Tensor) -> tuple[float, float, float, float]:
    rows, cols = gt.nonzero(as_tuple=True)
    return (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))


def evaluate(
    segmenter: Segmenter,
    images: torch.Tensor,
    trimaps: torch.Tensor,
    max_clicks: int = 5,
    automatic: bool = False,
    limit: int | None = None,
) -> dict:
    """images uint8 [N, 3, S, S], trimaps uint8 [N, S, S] -> aggregate metrics."""
    n = min(len(images), limit) if limit else len(images)
    click_ious = [[] for _ in range(max_clicks)]
    nocs, box_ious, box_bious, click1_bious, auto_ious = [], [], [], [], []
    t0 = time.time()
    used = 0
    for i in range(n):
        gt = trimaps[i] != 2
        if not gt.any():
            continue
        used += 1
        segmenter.set_image(images[i].permute(1, 2, 0).numpy())

        if automatic:
            auto_ious.append(float(metrics.iou(torch.from_numpy(segmenter.predict(Prompt()).mask), gt)))

        points, labels, ious = [], [], []
        pred = torch.zeros_like(gt)
        for k in range(max_clicks):
            click = (deepest_point(gt), POSITIVE) if k == 0 else next_click(pred, gt)
            if click is None:  # perfect already: carry the IoU forward
                ious.append(ious[-1])
                continue
            points.append(click[0])
            labels.append(click[1])
            pred = torch.from_numpy(segmenter.predict(Prompt(points, labels)).mask)
            ious.append(float(metrics.iou(pred, gt)))
            if k == 0:
                click1_bious.append(float(metrics.boundary_iou(pred, gt)))
        for k, v in enumerate(ious):
            click_ious[k].append(v)
        nocs.append(metrics.noc(ious, threshold=0.85, max_clicks=max_clicks))

        box_pred = torch.from_numpy(segmenter.predict(Prompt(box=tight_box(gt))).mask)
        box_ious.append(float(metrics.iou(box_pred, gt)))
        box_bious.append(float(metrics.boundary_iou(box_pred, gt)))
        if used % 100 == 0:
            logger.info("evaluated %d/%d images", used, n)

    result = {
        "backend": segmenter.backend.value,
        "images": used,
        "seconds_per_image": round((time.time() - t0) / max(1, used), 3),
        "iou_at_clicks": [round(float(np.mean(v)), 4) for v in click_ious],
        "noc85": round(float(np.mean(nocs)), 3),
        "iou_box": round(float(np.mean(box_ious)), 4),
        "boundary_iou_click1": round(float(np.mean(click1_bious)), 4),
        "boundary_iou_box": round(float(np.mean(box_bious)), 4),
    }
    if automatic:
        result["iou_automatic"] = round(float(np.mean(auto_ious)), 4)
    return result
