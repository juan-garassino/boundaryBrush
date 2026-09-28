"""Prompt and result types shared by every segmenter backend.

Coordinates are continuous pixel coordinates (x, y) in the image the prompt refers to:
pixel (row r, col c) covers [c, c+1) x [r, r+1), so a click on it is (c + 0.5, r + 0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

POSITIVE = 1
NEGATIVE = 0


@dataclass(frozen=True)
class Prompt:
    points: tuple[tuple[float, float], ...] = ()
    labels: tuple[int, ...] = ()
    box: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)

    def __post_init__(self) -> None:
        points = tuple(tuple(float(v) for v in p) for p in self.points)
        if any(len(p) != 2 for p in points):
            raise ValueError("points must be (x, y) pairs")
        labels = tuple(int(v) for v in self.labels)
        if len(points) != len(labels):
            raise ValueError(f"{len(points)} points but {len(labels)} labels")
        if any(v not in (POSITIVE, NEGATIVE) for v in labels):
            raise ValueError("labels must be 1 (include) or 0 (exclude)")
        coords = [v for p in points for v in p]
        box = None
        if self.box is not None:
            box = tuple(float(v) for v in self.box)
            if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("box must be (x0, y0, x1, y1) with x0 < x1 and y0 < y1")
            coords.extend(box)
        if any(not math.isfinite(v) or v < 0 for v in coords):
            raise ValueError("coordinates must be finite and non-negative")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "box", box)

    @property
    def is_empty(self) -> bool:
        return not self.points and self.box is None

    def check_within(self, width: int, height: int) -> None:
        xs = [p[0] for p in self.points] + ([self.box[0], self.box[2]] if self.box else [])
        ys = [p[1] for p in self.points] + ([self.box[1], self.box[3]] if self.box else [])
        if any(x > width for x in xs) or any(y > height for y in ys):
            raise ValueError(f"prompt falls outside the {width}x{height} image")

    def scaled(self, sx: float, sy: float) -> Prompt:
        box = None
        if self.box is not None:
            x0, y0, x1, y1 = self.box
            box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        return Prompt(tuple((x * sx, y * sy) for x, y in self.points), self.labels, box)


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray = field(repr=False)  # bool, H x W, same size as the image passed to set_image
    score: float

    def __post_init__(self) -> None:
        if self.mask.dtype != np.bool_ or self.mask.ndim != 2:
            raise ValueError(f"mask must be a 2-D bool array, got {self.mask.dtype} {self.mask.shape}")
