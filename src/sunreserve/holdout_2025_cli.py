"""Dedicated command entry point for the frozen 2025 holdout replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sunreserve.data import DataError, default_project_root
from sunreserve.holdout_2025 import evaluate_2025_holdout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sunreserve-holdout")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="project root (defaults to the checked-out repository containing this package)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    holdout = commands.add_parser(
        "evaluate-2025",
        help="apply a frozen development model to the separate untouched 2025 period once",
    )
    holdout.add_argument(
        "--manifest-sha256",
        required=True,
        help="exact validation selection-manifest digest inspected and frozen before scoring",
    )
    holdout.add_argument(
        "--model-sha256",
        required=True,
        help="exact training-only numeric model-bundle digest bound to the manifest",
    )
    holdout.add_argument(
        "--run-id",
        required=True,
        help="new immutable run ID under artifacts/runs; existing IDs are never overwritten",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = (args.root or default_project_root()).resolve()
        result = evaluate_2025_holdout(
            root,
            manifest_sha256=args.manifest_sha256,
            model_sha256=args.model_sha256,
            run_id=args.run_id,
        )
        print("Frozen model applied to 2025 targets only; no refit or selection was performed.")
        print(f"Eligible common targets: {result.sample_count}")
        print(f"Model: {result.model_id}")
        print(f"Holdout run: {result.run_dir}")
        return 0
    except (DataError, ValueError, OSError) as error:
        print(f"sunreserve-holdout: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
