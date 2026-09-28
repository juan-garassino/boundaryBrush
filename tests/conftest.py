"""Shared fixtures.

`synthetic_source` stands in for torchvision's OxfordIIITPet (data.py `PetSource` protocol):
each sample is an RGB image whose red channel is 255 exactly on the "pet" (a rectangle)
plus an Oxford-style trimap (1 pet, 2 background, 3 border band), so tests can check that
augmentation keeps image and mask aligned without downloading anything.
"""

from __future__ import annotations

import numpy as np
import pytest


class _SyntheticPets:
    def __init__(self, n: int = 4, h: int = 40, w: int = 56):
        self.samples = []
        rng = np.random.default_rng(0)
        for i in range(n):
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[..., 1:] = rng.integers(0, 120, size=(h, w, 2), dtype=np.uint8)
            tri = np.full((h, w), 2, dtype=np.uint8)
            top, left = 6 + i, 8 + 2 * i
            bh, bw = 20, 26
            tri[top : top + bh, left : left + bw] = 3
            tri[top + 1 : top + bh - 1, left + 1 : left + bw - 1] = 1
            img[..., 0] = np.where(tri != 2, 255, 0)
            self.samples.append((img, tri))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]


@pytest.fixture
def synthetic_source():
    return _SyntheticPets()
