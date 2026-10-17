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


@pytest.mark.parametrize("url", ["http://example.com/v1", "ftp://x/y", "example.com/v1", ""])
def test_release_url_must_be_https(monkeypatch, url):
    monkeypatch.setenv("BOUNDARYBRUSH_RELEASE_URL", url)
    with pytest.raises(ValueError, match="BOUNDARYBRUSH_RELEASE_URL"):
        get_settings()


@pytest.mark.parametrize("url", ["https://example.com/v2", "HTTPS://example.com/v2", "file:///srv/mirror/v1"])
def test_release_url_https_or_local_mirror_accepted(monkeypatch, url):
    # file:// stays allowed for offline mirrors: downloads are SHA-256 verified against the manifest anyway
    monkeypatch.setenv("BOUNDARYBRUSH_RELEASE_URL", url)
    assert get_settings().release_url == url


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:12", "mps"])
def test_device_accepted(monkeypatch, device):
    monkeypatch.setenv("BOUNDARYBRUSH_DEVICE", device)
    assert get_settings().device == device


@pytest.mark.parametrize("device", ["gpu", "auto", "CPU ", "cuda:", "cuda:x", ""])
def test_device_typo_rejected_at_startup(monkeypatch, device):
    monkeypatch.setenv("BOUNDARYBRUSH_DEVICE", device)
    with pytest.raises(ValueError, match="BOUNDARYBRUSH_DEVICE"):
        get_settings()


@pytest.mark.parametrize("name", ["DATA_DIR", "RUNS_DIR", "WEIGHTS_DIR"])
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_path_env_rejected_instead_of_meaning_cwd(monkeypatch, name, value):
    # Path("") is ".", so a blank var would silently put data/weights in the current directory
    monkeypatch.setenv(f"BOUNDARYBRUSH_{name}", value)
    with pytest.raises(ValueError, match=f"BOUNDARYBRUSH_{name}"):
        get_settings()
