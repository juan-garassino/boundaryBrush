from __future__ import annotations

import numpy as np
import torch

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import MaskResult
from boundarybrush.evaluate import evaluate, next_click, tight_box
from boundarybrush.providers.base import Segmenter


def _data(n=3, size=32):
    images = torch.zeros(n, 3, size, size, dtype=torch.uint8)
    trimaps = torch.full((n, size, size), 2, dtype=torch.uint8)
    for i in range(n):
        trimaps[i, 4 + i : 20 + i, 6 : 26 - i] = 1
        images[i, 0] = i  # lets the oracle recover which sample it was given
    return images, trimaps


class _Oracle(Segmenter):
    backend = Backend.MOCK

    def __init__(self, trimaps):
        super().__init__()
        self.trimaps = trimaps
        self.prompts = []

    def _embed(self, image):
        self.gt = (self.trimaps[int(image[0, 0, 0])] != 2).numpy()

    def _predict(self, prompt):
        self.prompts.append(prompt)
        return MaskResult(mask=self.gt.copy(), score=1.0)


class _Empty(_Oracle):
    def _predict(self, prompt):
        self.prompts.append(prompt)
        return MaskResult(mask=np.zeros_like(self.gt), score=0.0)


def test_oracle_scores_perfectly():
    images, trimaps = _data()
    r = evaluate(_Oracle(trimaps), images, trimaps, automatic=True)
    assert r["images"] == 3 and r["noc85"] == 1
    assert r["iou_at_clicks"] == [1.0] * 5 and r["iou_box"] == 1.0 and r["iou_automatic"] == 1.0


def test_empty_segmenter_uses_all_clicks():
    images, trimaps = _data()
    seg = _Empty(trimaps)
    r = evaluate(seg, images, trimaps)
    assert r["noc85"] == 5 and r["iou_at_clicks"] == [0.0] * 5
    clicks = [p for p in seg.prompts if p.box is None]
    assert all(set(p.labels) == {1} for p in clicks)  # only FN errors -> only positive clicks


def test_click_sequence_is_deterministic():
    images, trimaps = _data()
    a, b = _Empty(trimaps), _Empty(trimaps)
    evaluate(a, images, trimaps)
    evaluate(b, images, trimaps)
    assert a.prompts == b.prompts


def test_next_click_prefers_the_larger_error_and_labels_it():
    gt = torch.zeros(20, 20, dtype=torch.bool)
    gt[2:8, 2:8] = True
    pred = gt.clone()
    pred[10:19, 10:19] = True  # big false positive
    (x, y), label = next_click(pred, gt)
    assert label == 0 and pred[int(y), int(x)] and not gt[int(y), int(x)]
    assert next_click(gt, gt) is None


def test_tight_box_uses_pixel_edges():
    gt = torch.zeros(10, 10, dtype=torch.bool)
    gt[2:5, 3:7] = True
    assert tight_box(gt) == (3.0, 2.0, 7.0, 5.0)
