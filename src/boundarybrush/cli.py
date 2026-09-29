"""boundarybrush — promptable segmentation (U-Net, mini-SAM, SlimSAM) behind one predictor API.

Each subcommand imports its heavy modules lazily so `--help` stays fast.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from boundarybrush import __version__

logger = logging.getLogger(__name__)

BACKENDS = ["unet", "minisam", "slimsam", "mock"]


def parse_click(text: str) -> tuple[tuple[float, float], int]:
    """'x,y' or 'x,y:label' (label 1 include, 0 exclude)."""
    coords, _, label = text.partition(":")
    try:
        x, y = (float(v) for v in coords.split(","))
        lab = int(label) if label else 1
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad click {text!r}; expected x,y or x,y:0") from e
    if lab not in (0, 1):
        raise argparse.ArgumentTypeError(f"bad click label in {text!r}; use 0 or 1")
    return (x, y), lab


def parse_box(text: str) -> tuple[float, float, float, float]:
    try:
        x0, y0, x1, y1 = (float(v) for v in text.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"bad box {text!r}; expected x0,y0,x1,y1") from e
    return x0, y0, x1, y1


def _settings():
    import torch

    from boundarybrush.config import get_settings

    settings = get_settings()
    torch.set_num_threads(settings.num_threads)
    return settings


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    from boundarybrush import data

    settings = _settings()
    size = args.size or settings.image_size
    for split in ("trainval", "test"):
        path = data.cache_path(settings.data_dir, split, size)
        if path.exists() and not args.force:
            logger.info("cache exists: %s", path)
            continue
        source = data.OxfordPetSource(settings.data_dir, split, download=not args.no_download)
        logger.info("building %s cache (%d images) at %dpx", split, len(source), size)
        data.save_cache(data.build_cache(source, size), path)
        logger.info("wrote %s", path)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    from boundarybrush.train import TrainConfig, train

    settings = _settings()
    cfg = TrainConfig(
        kind=args.model,
        data_dir=str(settings.data_dir),
        out_dir=str(args.out_dir or settings.runs_dir),
        image_size=settings.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        limit=args.limit,
        prompts_per_image=args.prompts_per_image,
    )
    best = train(cfg, resume=args.resume)
    logger.info("best checkpoint: %s", best)
    return 0


def _segmenter(args, settings):
    from boundarybrush.providers.registry import get_segmenter

    return get_segmenter(args.backend, weights=args.weights, settings=settings)


def _cmd_eval(args: argparse.Namespace) -> int:
    from boundarybrush import data
    from boundarybrush.evaluate import evaluate

    settings = _settings()
    cache = data.load_cache(data.cache_path(settings.data_dir, "test", settings.image_size))
    limit = args.limit if args.limit is not None else (100 if args.backend == "slimsam" else None)
    result = evaluate(
        _segmenter(args, settings),
        cache["images"],
        cache["trimaps"],
        max_clicks=args.max_clicks,
        automatic=args.backend == "unet",
        limit=limit,
    )
    out = Path(args.out or f"results/eval_{args.backend}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from boundarybrush.domain.models import Prompt
    from boundarybrush.viz import load_image, overlay, save_png

    settings = _settings()
    image = load_image(args.image)
    prompt = Prompt(points=[c[0] for c in args.click], labels=[c[1] for c in args.click], box=args.box)
    seg = _segmenter(args, settings)
    seg.set_image(image)
    result = seg.predict(prompt)
    out = Path(args.out)
    save_png(result.mask.astype("uint8") * 255, out / "mask.png")
    save_png(overlay(image, result.mask, prompt), out / "overlay.png")
    print(json.dumps({"score": round(result.score, 4), "pixels": int(result.mask.sum()), "out": str(out)}))
    return 0


def _cmd_click(args: argparse.Namespace) -> int:  # pragma: no cover - GUI
    import matplotlib

    if args.mpl_backend:
        matplotlib.use(args.mpl_backend)
    from boundarybrush.viz import click_app, load_image

    settings = _settings()
    click_app(load_image(args.image), _segmenter(args, settings), out_dir=args.out)
    return 0


def _cmd_gallery(args: argparse.Namespace) -> int:
    import torch

    from boundarybrush import data
    from boundarybrush.domain.models import Prompt
    from boundarybrush.evaluate import tight_box
    from boundarybrush.ops import deepest_point
    from boundarybrush.providers.registry import get_segmenter
    from boundarybrush.viz import gallery

    settings = _settings()
    cache = data.load_cache(data.cache_path(settings.data_dir, "test", settings.image_size))
    segs = {b: get_segmenter(b, settings=settings) for b in args.backends.split(",")}
    gen = torch.Generator().manual_seed(args.seed)
    picks = torch.randperm(len(cache["images"]), generator=gen)[: args.n].tolist()
    rows = []
    for i in picks:
        image = cache["images"][i].permute(1, 2, 0).numpy()
        gt = cache["trimaps"][i] != 2
        click = Prompt(points=[deepest_point(gt)], labels=[1])
        box = Prompt(box=tight_box(gt))
        panels = {"ground truth": (gt.numpy(), None)}
        for name, seg in segs.items():
            seg.set_image(image)
            panels[f"{name} · 1 click"] = (seg.predict(click).mask, click)
            panels[f"{name} · box"] = (seg.predict(box).mask, box)
        rows.append({"image": image, "panels": panels})
    path = gallery(rows, args.out, title="Oxford-IIIT Pet test set (128px)")
    logger.info("wrote %s", path)
    return 0


def _cmd_fetch_weights(args: argparse.Namespace) -> int:
    from boundarybrush.weights import fetch_all

    settings = _settings()
    for path in fetch_all(settings.release_url, Path(args.dest) if args.dest else settings.weights_dir):
        logger.info("ready: %s", path)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from boundarybrush.checkpoint import export_weights
    from boundarybrush.weights import write_manifest

    paths = [
        export_weights(Path(src), Path(args.dest) / f"{Path(src).parent.name}.pt") for src in args.checkpoints
    ]
    if args.manifest:
        write_manifest(paths, Path(args.manifest))
    for p in paths:
        logger.info("exported %s", p)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boundarybrush", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare-data", help="download Oxford-IIIT Pet and build the resized tensor cache")
    p.add_argument("--size", type=int, default=None, help="cache resolution (default: settings.image_size)")
    p.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    p.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    p.set_defaults(func=_cmd_prepare_data)

    p = sub.add_parser("train", help="train the U-Net or the mini-SAM")
    p.add_argument("model", choices=["unet", "minisam"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="train on the first N images only")
    p.add_argument("--prompts-per-image", type=int, default=3)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--resume", action="store_true", help="continue from <out-dir>/<model>/last.pt")
    p.set_defaults(func=_cmd_train)

    def add_backend(p):
        p.add_argument("--backend", "-b", choices=BACKENDS, default="minisam")
        p.add_argument("--weights", default=None, help="checkpoint path (default: weights dir)")

    p = sub.add_parser("eval", help="interactive-segmentation metrics on the Pets test split")
    p.add_argument("backend", choices=BACKENDS)
    p.add_argument("--weights", default=None)
    p.add_argument("--limit", type=int, default=None, help="first N test images (slimsam: 100)")
    p.add_argument("--max-clicks", type=int, default=5)
    p.add_argument("--out", default=None, help="JSON path (default results/eval_<backend>.json)")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("predict", help="segment one image from clicks and/or a box")
    p.add_argument("image")
    add_backend(p)
    p.add_argument("--click", type=parse_click, action="append", default=[], help="x,y or x,y:0 (repeatable)")
    p.add_argument("--box", type=parse_box, default=None, help="x0,y0,x1,y1")
    p.add_argument("-o", "--out", default="out")
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("click", help="interactive click-to-segment window")
    p.add_argument("image")
    add_backend(p)
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--mpl-backend", default=None, help="e.g. TkAgg if the macOS window misbehaves")
    p.set_defaults(func=_cmd_click)

    p = sub.add_parser("gallery", help="README comparison figure on random test images")
    p.add_argument("--backends", default="unet,minisam,slimsam")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="docs/images/gallery.png")
    p.set_defaults(func=_cmd_gallery)

    p = sub.add_parser("fetch-weights", help="download released U-Net and mini-SAM weights (SHA-256 checked)")
    p.add_argument("--dest", default=None)
    p.set_defaults(func=_cmd_fetch_weights)

    p = sub.add_parser("export", help="strip training state from checkpoints for release")
    p.add_argument("checkpoints", nargs="+", help="e.g. runs/unet/best.pt runs/minisam/best.pt")
    p.add_argument("--dest", default="release")
    p.add_argument("--manifest", default=None, help="also write a SHA-256 manifest here")
    p.set_defaults(func=_cmd_export)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return args.func(args)
