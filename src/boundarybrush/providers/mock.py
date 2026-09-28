"""Deterministic segmenter for tests and demos without weights: discs around clicks, box fill."""

from __future__ import annotations

import numpy as np

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import POSITIVE, MaskResult, Prompt
from boundarybrush.providers.base import Segmenter


class MockSegmenter(Segmenter):
    backend = Backend.MOCK

    def _embed(self, image: np.ndarray) -> None:
        pass

    def _predict(self, prompt: Prompt) -> MaskResult:
        h, w = self.image_size
        yy, xx = np.mgrid[0:h, 0:w] + 0.5
        radius = min(h, w) / 8
        mask = np.zeros((h, w), dtype=bool)
        if prompt.box is not None:
            x0, y0, x1, y1 = prompt.box
            mask |= (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        for (x, y), label in zip(prompt.points, prompt.labels, strict=True):
            disc = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
            mask = mask | disc if label == POSITIVE else mask & ~disc
        return MaskResult(mask=mask, score=1.0)
