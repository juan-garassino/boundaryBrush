"""U-Net behind the predictor API: it segments everything; prompts pick connected components."""

from __future__ import annotations

import numpy as np
import torch

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import POSITIVE, MaskResult, Prompt
from boundarybrush.ops import component_at, logits_to_mask, resize_rgb, to_model_input
from boundarybrush.providers.base import Segmenter

_OFF = -1e4  # "definitely background" logit; finite so bilinear upsampling stays NaN-free


class UNetSegmenter(Segmenter):
    backend = Backend.UNET

    def __init__(self, model: torch.nn.Module, input_size: int = 128):
        super().__init__()
        self.model = model.eval()
        self.input_size = input_size
        self._logits: torch.Tensor | None = None

    @torch.inference_mode()
    def _embed(self, image: np.ndarray) -> None:
        x = to_model_input(resize_rgb(image, self.input_size))[None]
        self._logits = self.model(x)[0, 0]

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        h, w = self.image_size
        s = self.input_size
        return min(int(y * s / h), s - 1), min(int(x * s / w), s - 1)

    def _predict(self, prompt: Prompt) -> MaskResult:
        logits = self._logits
        full = logits > 0
        selected = full.clone()
        positives = [p for p, lab in zip(prompt.points, prompt.labels, strict=True) if lab == POSITIVE]
        negatives = [p for p, lab in zip(prompt.points, prompt.labels, strict=True) if lab != POSITIVE]
        if positives:
            selected = torch.zeros_like(full)
            for x, y in positives:
                selected |= component_at(full, *self._cell(x, y))
        for x, y in negatives:
            selected &= ~component_at(full, *self._cell(x, y))
        if prompt.box is not None:
            h, w = self.image_size
            s = self.input_size
            x0, y0, x1, y1 = prompt.box
            box = torch.zeros_like(full)
            box[int(y0 * s / h) : int(np.ceil(y1 * s / h)), int(x0 * s / w) : int(np.ceil(x1 * s / w))] = True
            selected &= box
        mask = logits_to_mask(torch.where(selected, logits, torch.full_like(logits, _OFF)), self.image_size)
        score = float(torch.sigmoid(logits[selected]).mean()) if selected.any() else 0.0
        return MaskResult(mask=mask, score=score)
