"""The from-scratch mini-SAM behind the predictor API."""

from __future__ import annotations

import numpy as np
import torch

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import MaskResult, Prompt
from boundarybrush.ops import logits_to_mask, resize_rgb, to_model_input
from boundarybrush.prompts import encode_prompt
from boundarybrush.providers.base import Segmenter


class MiniSAMSegmenter(Segmenter):
    backend = Backend.MINISAM

    def __init__(self, model: torch.nn.Module, input_size: int = 128):
        super().__init__()
        self.model = model.eval()
        self.input_size = input_size
        self._feats: dict | None = None

    @torch.inference_mode()
    def _embed(self, image: np.ndarray) -> None:
        self._feats = self.model.encode_image(to_model_input(resize_rgb(image, self.input_size))[None])

    @torch.inference_mode()
    def _predict(self, prompt: Prompt) -> MaskResult:
        if prompt.is_empty:
            raise ValueError("mini-SAM needs at least one click or a box")
        h, w = self.image_size
        coords, types = encode_prompt(prompt, width=w, height=h)  # normalized, so squashing is harmless
        masks, iou_pred = self.model.decode(self._feats, coords[None], types[None])
        best = int(iou_pred[0].argmax())
        mask = logits_to_mask(masks[0, best], (h, w))
        return MaskResult(mask=mask, score=float(iou_pred[0, best].clamp(0, 1)))
