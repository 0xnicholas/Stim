"""Command-line entry point for the Stim service."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from stim_serve import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stim-serve",
        description="Serve Stim decision models over the System One protocol.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
