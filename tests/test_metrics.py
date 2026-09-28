from __future__ import annotations

import pytest
import torch

from boundarybrush import metrics


def _box(h, w, top, left, bh, bw):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[top : top + bh, left : left + bw] = True
    return m


def test_iou_identical_disjoint_half():
    a = _box(10, 10, 0, 0, 4, 4)
    assert metrics.iou(a, a).item() == 1.0
    assert metrics.iou(a, _box(10, 10, 6, 6, 4, 4)).item() == 0.0
    half = _box(10, 10, 0, 0, 4, 2)  # 8 px inside a's 16
    assert metrics.iou(half, a).item() == pytest.approx(0.5)


def test_iou_both_empty_is_one_and_is_per_image():
    empty = torch.zeros(2, 5, 5, dtype=torch.bool)
    assert metrics.iou(empty, empty).tolist() == [1.0, 1.0]


def test_boundary_iou_identical_and_shifted():
    a = _box(64, 64, 10, 10, 30, 30)
    assert metrics.boundary_iou(a, a).item() == 1.0
    shifted = _box(64, 64, 14, 14, 30, 30)
    assert metrics.boundary_iou(shifted, a).item() < metrics.iou(shifted, a).item()


def test_boundary_band_width_scales_with_diagonal():
    assert metrics.boundary_width(128, 128) == 4
    assert metrics.boundary_width(8, 8) == 1


def test_noc():
    assert metrics.noc([0.9, 0.95], threshold=0.85, max_clicks=5) == 1
    assert metrics.noc([0.5, 0.7, 0.86], threshold=0.85, max_clicks=5) == 3
    assert metrics.noc([0.1, 0.2, 0.3, 0.4, 0.5], threshold=0.85, max_clicks=5) == 5
