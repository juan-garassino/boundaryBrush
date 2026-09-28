"""The predictor interface every backend implements (SAM-style: embed once, prompt many times)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import MaskResult, Prompt


class Segmenter(ABC):
    backend: Backend

    def __init__(self) -> None:
        self._hw: tuple[int, int] | None = None

    @property
    def image_size(self) -> tuple[int, int]:
        """(height, width) of the image passed to set_image."""
        if self._hw is None:
            raise RuntimeError("call set_image() before predict()")
        return self._hw

    def set_image(self, image: np.ndarray) -> None:
        """image: RGB uint8 H x W x 3. Computes and caches whatever the backend needs."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an RGB uint8 H x W x 3 image, got {image.dtype} {image.shape}")
        self._hw = (image.shape[0], image.shape[1])
        self._embed(image)

    def predict(self, prompt: Prompt) -> MaskResult:
        h, w = self.image_size
        prompt.check_within(width=w, height=h)
        return self._predict(prompt)

    @abstractmethod
    def _embed(self, image: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, prompt: Prompt) -> MaskResult: ...
