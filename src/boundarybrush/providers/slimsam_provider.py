"""Pretrained SlimSAM (a pruned SAM-B, 9.7M params) via transformers — optional extra `[sam]`.

The image embedding is computed once in set_image (the slow part on CPU); predict only runs
the prompt encoder + mask decoder. Coordinates are scaled here rather than by SamProcessor,
which would re-run image preprocessing on every call.
"""

from __future__ import annotations

import numpy as np
import torch

from boundarybrush.domain.enums import Backend
from boundarybrush.domain.models import MaskResult, Prompt
from boundarybrush.providers.base import Segmenter


class SlimSAMSegmenter(Segmenter):
    backend = Backend.SLIMSAM

    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor
        self._cache: dict | None = None

    @classmethod
    def from_pretrained(cls, model_id: str) -> SlimSAMSegmenter:
        try:
            from transformers import SamModel, SamProcessor
        except ImportError as e:  # pragma: no cover - depends on the environment
            raise ImportError("SlimSAM needs the [sam] extra: uv sync --extra sam") from e
        model = SamModel.from_pretrained(model_id, use_safetensors=True).eval()
        return cls(model, SamProcessor.from_pretrained(model_id))

    @torch.inference_mode()
    def _embed(self, image: np.ndarray) -> None:
        inputs = self.processor(images=image, return_tensors="pt")
        self._cache = {
            "embeddings": self.model.get_image_embeddings(inputs["pixel_values"]),
            "original_sizes": inputs["original_sizes"],
            "reshaped_input_sizes": inputs["reshaped_input_sizes"],
        }

    def _scale(self) -> float:
        h, w = self.image_size
        return self.processor.image_processor.size["longest_edge"] / max(h, w)

    @torch.inference_mode()
    def _predict(self, prompt: Prompt) -> MaskResult:
        if prompt.is_empty:
            raise ValueError("SlimSAM needs at least one click or a box")
        s = self._scale()
        kwargs = {"image_embeddings": self._cache["embeddings"]}
        if prompt.points:
            kwargs["input_points"] = torch.tensor([[[[x * s, y * s] for x, y in prompt.points]]])
            kwargs["input_labels"] = torch.tensor([[list(prompt.labels)]])
        if prompt.box is not None:
            kwargs["input_boxes"] = torch.tensor([[[v * s for v in prompt.box]]])
        multimask = len(prompt.points) == 1 and prompt.box is None
        out = self.model(**kwargs, multimask_output=multimask)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks, self._cache["original_sizes"], self._cache["reshaped_input_sizes"]
        )[0][0]  # (n_masks, H, W) bool
        scores = out.iou_scores[0, 0]
        best = int(scores.argmax())
        return MaskResult(mask=masks[best].numpy().astype(bool), score=float(scores[best]))
