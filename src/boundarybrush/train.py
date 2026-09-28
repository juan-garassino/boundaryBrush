"""Training loop for the U-Net and the mini-SAM on the cached Pets tensors (CPU-first)."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from boundarybrush import data, prompts
from boundarybrush.checkpoint import build_model, load_checkpoint, save_checkpoint
from boundarybrush.models.losses import hard_iou, minisam_loss, weighted_bce_dice
from boundarybrush.ops import deepest_point

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    kind: str  # "unet" | "minisam"
    data_dir: str = "data"
    out_dir: str = "runs"
    image_size: int = 128
    epochs: int = 40
    batch_size: int = 32
    lr: float | None = None  # default per model: 1e-3 unet, 3e-4 minisam
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    seed: int = 0
    val_fraction: float = 0.1
    band_weight: float = 0.5
    prompts_per_image: int = 3
    paste_prob: float = 0.5
    limit: int | None = None  # use only the first N training images (smoke runs)
    log_every: int = 20
    model_config: dict = field(default_factory=dict)

    @property
    def learning_rate(self) -> float:
        return self.lr or (1e-3 if self.kind == "unet" else 3e-4)


class MetricsWriter:
    """Append-only JSONL, one object per line, flushed per write."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a")

    def log(self, **fields) -> None:
        fields.setdefault("ts", time.time())
        self._file.write(json.dumps(fields) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _loaders(cfg: TrainConfig, cache: dict) -> tuple[DataLoader, data.PetsDataset]:
    train_idx, val_idx = data.split_indices(len(cache["images"]), cfg.val_fraction, cfg.seed)
    if cfg.limit:
        train_idx, val_idx = train_idx[: cfg.limit], val_idx[: max(2, cfg.limit // 10)]
    common = dict(seed=cfg.seed, band_weight=cfg.band_weight)
    if cfg.kind == "minisam":
        train_ds = data.PromptedPetsDataset(
            cache,
            train_idx,
            augment=True,
            prompts_per_image=cfg.prompts_per_image,
            paste_prob=cfg.paste_prob,
            **common,
        )
    else:
        train_ds = data.PetsDataset(cache, train_idx, augment=True, **common)
    val_ds = data.PetsDataset(cache, val_idx, augment=False, **common)
    loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    return loader, val_ds


def _step(model, kind: str, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
    if kind == "unet":
        logits = model(batch["image"])[:, 0]
        loss = weighted_bce_dice(logits, batch["target"], batch["weight"]).mean()
        return loss, {"iou": float(hard_iou(logits, batch["target"]).mean())}
    masks, iou_pred = model(batch["image"], batch["coords"], batch["types"])
    return minisam_loss(
        masks.flatten(0, 1),
        iou_pred.flatten(0, 1),
        batch["target"].flatten(0, 1),
        batch["weight"].flatten(0, 1),
    )


def box_prompt(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Tight ground-truth box as prompt tensors (the IoU@box protocol)."""
    h, w = target.shape
    rows, cols = target.nonzero(as_tuple=True)
    tokens = [
        (cols.min().item() / w, rows.min().item() / h, prompts.BOX_TL),
        ((cols.max().item() + 1) / w, (rows.max().item() + 1) / h, prompts.BOX_BR),
    ]
    return prompts._pad(tokens)


def click_prompt(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One positive click at the deepest point of the target (first click of the eval protocol)."""
    h, w = target.shape
    x, y = deepest_point(target)
    return prompts._pad([(x / w, y / h, prompts.POS)])


@torch.no_grad()
def validate(model, kind: str, val_ds: data.PetsDataset, batch_size: int = 64) -> dict[str, float]:
    model.eval()
    ious, box_ious = [], []
    for start in range(0, len(val_ds), batch_size):
        items = [val_ds[i] for i in range(start, min(start + batch_size, len(val_ds)))]
        images = torch.stack([it["image"] for it in items])
        targets = torch.stack([it["target"] for it in items])
        if kind == "unet":
            ious.append(hard_iou(model(images)[:, 0], targets))
            continue
        nonempty = targets.flatten(1).any(1)
        images, targets = images[nonempty], targets[nonempty]
        feats = model.encode_image(images)
        for store, make in ((ious, click_prompt), (box_ious, box_prompt)):
            coords, types = zip(*(make(t.bool()) for t in targets), strict=True)
            masks, iou_pred = model.decode(feats, torch.stack(coords), torch.stack(types))
            best = masks[torch.arange(len(masks)), iou_pred.argmax(1)]
            store.append(hard_iou(best, targets))
    model.train()
    out = {"val_iou": float(torch.cat(ious).mean())}
    if box_ious:
        out["val_box_iou"] = float(torch.cat(box_ious).mean())
        out["val_score"] = (out["val_iou"] + out["val_box_iou"]) / 2
    else:
        out["val_score"] = out["val_iou"]
    return out


def train(cfg: TrainConfig, resume: bool = False) -> Path:
    torch.manual_seed(cfg.seed)
    run_dir = Path(cfg.out_dir) / cfg.kind
    cache = data.load_cache(data.cache_path(Path(cfg.data_dir), "trainval", cfg.image_size))
    loader, val_ds = _loaders(cfg, cache)

    model = build_model(cfg.kind, cfg.model_config)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: warmup_cosine(s, min(cfg.warmup_steps, total_steps // 10 + 1), total_steps)
    )
    start_epoch, best = 1, -1.0
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if resume and last_path.exists():
        state = load_checkpoint(last_path)
        model.load_state_dict(state["state_dict"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state["best"]
        logger.info("resumed from epoch %d (best %.4f)", state["epoch"], best)

    writer = MetricsWriter(run_dir / "metrics.jsonl")
    (run_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "%s: %.2fM params, %d train batches/epoch, %d val images",
        cfg.kind,
        n_params / 1e6,
        len(loader),
        len(val_ds),
    )
    model.train()
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0, running = time.time(), []
        for i, batch in enumerate(loader, start=1):
            loss, stats = _step(model, cfg.kind, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running.append(float(loss))
            if i % cfg.log_every == 0:
                logger.info(
                    "epoch %d batch %d/%d loss %.4f", epoch, i, len(loader), sum(running) / len(running)
                )
                writer.log(epoch=epoch, batch=i, loss=float(loss), lr=sched.get_last_lr()[0], **stats)
        val = validate(model, cfg.kind, val_ds)
        epoch_stats = {
            "epoch": epoch,
            "train_loss": sum(running) / max(1, len(running)),
            "seconds": round(time.time() - t0, 1),
            **val,
        }
        writer.log(**epoch_stats)
        logger.info("epoch %d done: %s", epoch, epoch_stats)
        extra = {
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "epoch": epoch,
            "metrics": epoch_stats,
        }
        if val["val_score"] > best:
            best = val["val_score"]
            save_checkpoint(best_path, kind=cfg.kind, model=model, best=best, **extra)
        save_checkpoint(last_path, kind=cfg.kind, model=model, best=best, **extra)
    writer.close()
    return best_path
