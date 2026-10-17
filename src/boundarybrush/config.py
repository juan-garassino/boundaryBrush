"""Runtime settings read from BOUNDARYBRUSH_* environment variables.

A plain dataclass instead of pydantic-settings keeps the dependency list short.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

_RELEASE = "https://github.com/juan-garassino/boundaryBrush/releases/download/v1.0.0"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"BOUNDARYBRUSH_{name}", default)


_DEVICE_RE = re.compile(r"cpu|mps|cuda(:\d+)?")


def _str(name: str, default: str) -> str:
    return field(default_factory=lambda: _env(name, default))


def _positive_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"BOUNDARYBRUSH_{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"BOUNDARYBRUSH_{name} must be a positive integer, got {value}")
    return value


def _int(name: str, default: int) -> int:
    return field(default_factory=lambda: _positive_int(name, default))


def _release_url(name: str, default: str) -> str:
    # weights are fetched from here: plain http/ftp would let a network attacker swap checkpoints;
    # file:// stays allowed for offline mirrors on the local disk
    raw = _env(name, default)
    if not raw.lower().startswith(("https://", "file://")):
        raise ValueError(f"BOUNDARYBRUSH_{name} must be an https:// (or file://) URL, got {raw!r}")
    return raw


def _url(name: str, default: str) -> str:
    return field(default_factory=lambda: _release_url(name, default))


def _checked_device(name: str, default: str) -> str:
    raw = _env(name, default)
    if not _DEVICE_RE.fullmatch(raw):
        raise ValueError(f"BOUNDARYBRUSH_{name} must be cpu, mps, cuda or cuda:N, got {raw!r}")
    return raw


def _device(name: str, default: str) -> str:
    return field(default_factory=lambda: _checked_device(name, default))


def _path(name: str, default: str) -> Path:
    return field(default_factory=lambda: Path(_env(name, default)).expanduser())


@dataclass(frozen=True)
class Settings:
    data_dir: Path = _path("DATA_DIR", "data")  # BOUNDARYBRUSH_DATA_DIR=data
    runs_dir: Path = _path("RUNS_DIR", "runs")  # BOUNDARYBRUSH_RUNS_DIR=runs
    weights_dir: Path = _path("WEIGHTS_DIR", "~/.cache/boundarybrush")  # BOUNDARYBRUSH_WEIGHTS_DIR=~/w
    device: str = _device("DEVICE", "cpu")  # BOUNDARYBRUSH_DEVICE=cpu (never auto: MPS on Iris Pro)
    num_threads: int = _int("NUM_THREADS", 4)  # BOUNDARYBRUSH_NUM_THREADS=4 (physical cores)
    image_size: int = _int("IMAGE_SIZE", 128)  # BOUNDARYBRUSH_IMAGE_SIZE=128
    sam_model_id: str = _str("SAM_MODEL_ID", "nielsr/slimsam-77-uniform")  # BOUNDARYBRUSH_SAM_MODEL_ID
    release_url: str = _url("RELEASE_URL", _RELEASE)  # BOUNDARYBRUSH_RELEASE_URL=https://…/v1.0.0


def get_settings() -> Settings:
    return Settings()
