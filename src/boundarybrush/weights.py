"""Download released weights and verify them against the SHA-256 manifest shipped in the package."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from importlib import resources
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST = "weights_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, str]:
    text = resources.files("boundarybrush").joinpath(MANIFEST).read_text()
    return json.loads(text)


def fetch(name: str, base_url: str, dest_dir: Path, expected_sha: str) -> Path:
    """Fetch base_url/name into dest_dir, verifying the SHA-256 before it replaces anything."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and sha256(dest) == expected_sha:
        logger.info("%s already present and verified", dest)
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{base_url.rstrip('/')}/{name}"
    logger.info("downloading %s", url)
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as f:  # noqa: S310 - fixed https/file URLs
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    actual = sha256(tmp)
    if actual != expected_sha:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {name}: expected {expected_sha}, got {actual}")
    os.replace(tmp, dest)
    return dest


def fetch_all(base_url: str, dest_dir: Path, manifest: dict[str, str] | None = None) -> list[Path]:
    manifest = manifest if manifest is not None else load_manifest()
    if not manifest:
        raise RuntimeError("no released weights listed in the manifest yet")
    return [fetch(name, base_url, dest_dir, sha) for name, sha in manifest.items()]


def write_manifest(paths: list[Path], out: Path) -> Path:
    out.write_text(json.dumps({p.name: sha256(p) for p in paths}, indent=2) + "\n")
    return out
