from __future__ import annotations

import numpy as np
import pytest

from boundarybrush import weights
from boundarybrush.domain.models import Prompt
from boundarybrush.viz import gallery, overlay


def test_fetch_verifies_checksum(tmp_path):
    src = tmp_path / "release"
    src.mkdir()
    (src / "unet.pt").write_bytes(b"weights")
    good = weights.sha256(src / "unet.pt")
    dest = tmp_path / "cache"
    path = weights.fetch("unet.pt", src.as_uri(), dest, good)
    assert path.read_bytes() == b"weights"
    with pytest.raises(ValueError):
        weights.fetch("unet.pt", src.as_uri(), tmp_path / "other", "0" * 64)
    assert not (tmp_path / "other" / "unet.pt").exists()


def test_fetch_all_requires_a_manifest(tmp_path):
    with pytest.raises(RuntimeError):
        weights.fetch_all("file:///nowhere", tmp_path, manifest={})


def test_manifest_roundtrip(tmp_path):
    f = tmp_path / "a.pt"
    f.write_bytes(b"x")
    out = weights.write_manifest([f], tmp_path / "m.json")
    assert weights.sha256(f) in out.read_text()


def test_overlay_draws_mask_clicks_and_box():
    img = np.zeros((40, 60, 3), dtype=np.uint8)
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 10:30] = True
    out = overlay(img, mask, Prompt(points=[(50, 5)], labels=[0], box=(5, 5, 35, 35)))
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out[20, 20, 0] > 100  # tinted
    assert tuple(out[5, 50]) == (230, 40, 40)  # negative click


def test_gallery_writes_png(tmp_path):
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=bool)
    path = gallery([{"image": img, "panels": {"a": (mask, None)}}], tmp_path / "g.png", title="t")
    assert path.exists() and path.stat().st_size > 0
