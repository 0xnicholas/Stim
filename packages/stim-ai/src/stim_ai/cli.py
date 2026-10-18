"""Command-line entry point for the Stim model core."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from stim_ai import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stim",
        description="Stim decision-model command-line interface.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0
