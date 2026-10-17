"""argparse CLI; each subcommand imports its heavy modules lazily."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from typing import cast

from boundarybrush import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boundarybrush", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_subparsers(dest="cmd", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = cast("Callable[[argparse.Namespace], int] | None", getattr(args, "func", None))
    if func is None:  # a subparser registered without set_defaults(func=...)
        parser.error(f"subcommand {args.cmd!r} has no handler")
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return func(args)
