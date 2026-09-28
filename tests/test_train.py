from __future__ import annotations

import json
import math

import pytest
import torch

from boundarybrush import data
from boundarybrush.checkpoint import export_weights, load_model
from boundarybrush.train import TrainConfig, train, warmup_cosine

TINY = {
    "unet": {"base_channels": 4, "depth": 2},
    "minisam": {
        "embed_dim": 16,
        "n_heads": 2,
        "stem_channels": 4,
        "skip_channels": 4,
        "n_decoder_layers": 1,
        "n_image_attn_layers": 1,
    },
}


@pytest.fixture
def data_dir(tmp_path, synthetic_source):
    cache = data.build_cache(synthetic_source, size=16)
    both = {k: torch.cat([v, v, v]) for k, v in cache.items()}  # 12 images so val is non-empty
    data.save_cache(both, data.cache_path(tmp_path / "data", "trainval", 16))
    return tmp_path


def _cfg(kind, root, epochs):
    return TrainConfig(
        kind=kind,
        data_dir=str(root / "data"),
        out_dir=str(root / "runs"),
        image_size=16,
        epochs=epochs,
        batch_size=4,
        prompts_per_image=2,
        log_every=1,
        model_config=TINY[kind],
    )


def test_warmup_cosine_shape():
    assert warmup_cosine(0, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(10, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0)


@pytest.mark.parametrize("kind", ["unet", "minisam"])
def test_train_smoke_writes_checkpoints_and_resumes(kind, data_dir):
    best = train(_cfg(kind, data_dir, epochs=2))
    run = data_dir / "runs" / kind
    assert best.exists() and (run / "last.pt").exists()
    lines = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all(math.isfinite(line.get("loss", line.get("train_loss", 0.0))) for line in lines)
    assert [line["epoch"] for line in lines if "val_score" in line] == [1, 2]

    loaded_kind, model = load_model(best)
    assert loaded_kind == kind and not model.training

    train(_cfg(kind, data_dir, epochs=3), resume=True)
    epochs = [
        json.loads(line)["epoch"]
        for line in (run / "metrics.jsonl").read_text().splitlines()
        if "val_score" in json.loads(line)
    ]
    assert epochs == [1, 2, 3]

    exported = export_weights(best, data_dir / f"{kind}.pt")
    payload = torch.load(exported, weights_only=True)
    assert "optimizer" not in payload and payload["kind"] == kind
