from __future__ import annotations

from pathlib import Path

import pytest

from boundarybrush.config import get_settings


def test_defaults(monkeypatch):
    for name in ("DATA_DIR", "DEVICE", "NUM_THREADS", "IMAGE_SIZE"):
        monkeypatch.delenv(f"BOUNDARYBRUSH_{name}", raising=False)
    s = get_settings()
    assert s.data_dir == Path("data")
    assert s.device == "cpu"
    assert s.num_threads == 4
    assert s.image_size == 128


def test_env_override(monkeypatch):
    monkeypatch.setenv("BOUNDARYBRUSH_DEVICE", "cuda")
    monkeypatch.setenv("BOUNDARYBRUSH_IMAGE_SIZE", "64")
    s = get_settings()
    assert s.device == "cuda"
    assert s.image_size == 64


def test_bad_int_env_names_the_variable(monkeypatch):
    monkeypatch.setenv("BOUNDARYBRUSH_NUM_THREADS", "four")
    with pytest.raises(ValueError, match="BOUNDARYBRUSH_NUM_THREADS"):
        get_settings()


@pytest.mark.parametrize("value", ["0", "-2"])
def test_non_positive_int_env_rejected(monkeypatch, value):
    monkeypatch.setenv("BOUNDARYBRUSH_IMAGE_SIZE", value)
    with pytest.raises(ValueError, match="BOUNDARYBRUSH_IMAGE_SIZE"):
        get_settings()
