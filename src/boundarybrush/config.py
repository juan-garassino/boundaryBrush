"""Runtime settings read from BOUNDARYBRUSH_* environment variables.

A plain dataclass instead of pydantic-settings keeps the dependency list short.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_RELEASE = "https://github.com/juan-garassino/boundaryBrush/releases/download/v1.0.0"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"BOUNDARYBRUSH_{name}", default)


def _str(name: str, default: str):
    return field(default_factory=lambda: _env(name, default))


def _int(name: str, default: int):
    return field(default_factory=lambda: int(_env(name, str(default))))


def _path(name: str, default: str):
    return field(default_factory=lambda: Path(_env(name, default)).expanduser())


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _path("DATA_DIR", "data")  # BOUNDARYBRUSH_DATA_DIR=data
    runs_dir: Path = _path("RUNS_DIR", "runs")  # BOUNDARYBRUSH_RUNS_DIR=runs
    weights_dir: Path = _path("WEIGHTS_DIR", "~/.cache/boundarybrush")  # BOUNDARYBRUSH_WEIGHTS_DIR=~/w
    device: str = _str("DEVICE", "cpu")  # BOUNDARYBRUSH_DEVICE=cpu (never auto: MPS on Iris Pro)
    num_threads: int = _int("NUM_THREADS", 4)  # BOUNDARYBRUSH_NUM_THREADS=4 (physical cores)
    image_size: int = _int("IMAGE_SIZE", 128)  # BOUNDARYBRUSH_IMAGE_SIZE=128
    sam_model_id: str = _str("SAM_MODEL_ID", "nielsr/slimsam-77-uniform")  # BOUNDARYBRUSH_SAM_MODEL_ID
    release_url: str = _str("RELEASE_URL", _RELEASE)  # BOUNDARYBRUSH_RELEASE_URL=https://…/v1.0.0


def get_settings() -> Settings:
    return Settings()
