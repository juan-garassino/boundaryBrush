from __future__ import annotations

import numpy as np
import torch

from boundarybrush import ops


def _square(h=16, w=16, top=4, left=4, size=8):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[top : top + size, left : left + size] = True
    return m


def test_erode_shrinks_interior_square_by_one_each_side():
    e = ops.erode(_square(), 1)
    assert e.sum() == 6 * 6
    assert e[5:11, 5:11].all()


def test_erode_treats_image_border_as_boundary():
    m = torch.ones(8, 8, dtype=torch.bool)  # fills the frame
    e = ops.erode(m, 1)
    assert e.sum() == 6 * 6
    assert not e[0].any() and not e[:, 0].any()


def test_dilate_grows_square():
    d = ops.dilate(_square(), 1)
    assert d.sum() == 10 * 10


def test_erode_batched_shape():
    m = torch.stack([_square(), _square(top=0, left=0)])
    assert ops.erode(m, 1).shape == (2, 16, 16)


def test_depth_map_counts_erosions():
    depth = ops.depth_map(_square())
    assert depth.max() == 4  # 8x8 square survives 3 erosions -> depth 4 at the centre
    assert depth[0, 0] == 0


def test_deepest_point_is_pixel_centred_centre():
    x, y = ops.deepest_point(_square())
    assert 7.5 <= x <= 8.5 and 7.5 <= y <= 8.5


def test_deepest_point_on_ring_avoids_the_hole():
    m = _square(size=10, top=3, left=3)
    m[6:10, 6:10] = False
    x, y = ops.deepest_point(m)
    assert m[int(y), int(x)]


def test_deepest_point_empty_is_none():
    assert ops.deepest_point(torch.zeros(8, 8, dtype=torch.bool)) is None


def test_deepest_point_is_deterministic_on_ties():
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[1, 1] = m[5, 5] = True
    assert ops.deepest_point(m) == ops.deepest_point(m) == (1.5, 1.5)


def test_component_at_selects_one_blob():
    m = torch.zeros(16, 16, dtype=torch.bool)
    m[1:4, 1:4] = True
    m[10:14, 10:14] = True
    comp = ops.component_at(m, row=11, col=11)
    assert comp.sum() == 16 and comp[10:14, 10:14].all()


def test_component_at_background_is_empty():
    assert ops.component_at(_square(), row=0, col=0).sum() == 0


def test_resize_rgb_and_trimap():
    img = (np.random.default_rng(0).random((30, 50, 3)) * 255).astype(np.uint8)
    out = ops.resize_rgb(img, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    tri = np.full((30, 50), 2, dtype=np.uint8)
    tri[10:20, 10:40] = 1
    small = ops.resize_labels(tri, 16)
    assert small.shape == (16, 16) and set(np.unique(small)) <= {1, 2}


def test_to_model_input_normalizes():
    img = np.full((8, 8, 3), 128, dtype=np.uint8)
    x = ops.to_model_input(img)
    assert x.shape == (3, 8, 8) and x.dtype == torch.float32
    assert abs(float(x.mean())) < 1.0


def test_logits_to_mask_upsamples():
    logits = torch.full((4, 4), -5.0)
    logits[1:3, 1:3] = 5.0
    mask = ops.logits_to_mask(logits, (8, 12))
    assert mask.shape == (8, 12) and mask.dtype == np.bool_
    assert mask[4, 6] and not mask[0, 0]
