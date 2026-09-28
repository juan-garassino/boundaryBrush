"""get_segmenter(): the one entry point other projects (illusionFrame) use."""

from __future__ import annotations

import logging
from pathlib import Path

from boundarybrush.config import Settings, get_settings
from boundarybrush.domain.enums import Backend
from boundarybrush.providers.base import Segmenter

logger = logging.getLogger(__name__)


def default_weights(backend: Backend, settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return settings.weights_dir / f"{backend.value}.pt"


def get_segmenter(
    backend: Backend | str, weights: Path | str | None = None, settings: Settings | None = None
) -> Segmenter:
    settings = settings or get_settings()
    backend = Backend(backend)
    if backend is Backend.MOCK:
        from boundarybrush.providers.mock import MockSegmenter

        return MockSegmenter()
    if backend is Backend.SLIMSAM:
        from boundarybrush.providers.slimsam_provider import SlimSAMSegmenter

        return SlimSAMSegmenter.from_pretrained(settings.sam_model_id)

    from boundarybrush.checkpoint import load_model

    path = Path(weights) if weights else default_weights(backend, settings)
    if not path.exists():
        raise FileNotFoundError(
            f"no {backend} weights at {path}; run `boundarybrush fetch-weights` or train one"
        )
    kind, model = load_model(path)
    if kind != backend.value:
        raise ValueError(f"{path} holds a {kind} checkpoint, not {backend}")
    logger.info("loaded %s from %s", kind, path)
    if backend is Backend.UNET:
        from boundarybrush.providers.unet_provider import UNetSegmenter

        return UNetSegmenter(model, settings.image_size)
    from boundarybrush.providers.minisam_provider import MiniSAMSegmenter

    return MiniSAMSegmenter(model, settings.image_size)
