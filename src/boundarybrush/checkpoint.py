"""Checkpoints that rebuild their own model: architecture config travels with the weights.

Payloads hold only tensors and plain str/int/float/dict/list values, so every load uses
`torch.load(weights_only=True)` — safe for weights downloaded from a release.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn as nn

from boundarybrush import __version__
from boundarybrush.models.minisam import MiniSAM, MiniSAMConfig
from boundarybrush.models.unet import UNet, UNetConfig

logger = logging.getLogger(__name__)

MODELS = {"unet": (UNet, UNetConfig), "minisam": (MiniSAM, MiniSAMConfig)}


def build_model(kind: str, config: dict | None = None) -> nn.Module:
    model_cls, config_cls = MODELS[kind]
    return model_cls(config_cls(**(config or {})))


def save_checkpoint(path: Path, *, kind: str, model: nn.Module, **extra) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "torch_version": str(torch.__version__),
        "boundarybrush_version": __version__,
        **extra,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(path: Path) -> tuple[str, nn.Module]:
    payload = load_checkpoint(path)
    model = build_model(payload["kind"], payload["config"])
    model.load_state_dict(payload["state_dict"])
    return payload["kind"], model.eval()


def export_weights(src: Path, dst: Path) -> Path:
    """Strip optimizer/scheduler state for release: kind + config + weights only."""
    payload = load_checkpoint(src)
    kind, model = payload["kind"], build_model(payload["kind"], payload["config"])
    model.load_state_dict(payload["state_dict"])
    return save_checkpoint(dst, kind=kind, model=model, metrics=payload.get("metrics", {}))
