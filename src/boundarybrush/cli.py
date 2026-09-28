"""boundarybrush — promptable segmentation (U-Net, mini-SAM, SlimSAM) behind one predictor API.

Each subcommand imports its heavy modules lazily so `--help` stays fast.
"""

from __future__ import annotations

import argparse
import logging

from boundarybrush import __version__

logger = logging.getLogger(__name__)


def _cmd_prepare_data(args: argparse.Namespace) -> int:
    from boundarybrush import data
    from boundarybrush.config import get_settings

    settings = get_settings()
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return args.func(args)
