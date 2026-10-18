"""Command-line entry point for the System One protocol layer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from stim_systemone import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stim-systemone",
        description="Inspect the Stim System One protocol layer.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
