#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("BRUTEFIT_HEADLESS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/brutefit-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/brutefit-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

# Import-order note:
# PySide/shiboken on macOS has triggered fragile import hooks around dateutil/six in
# this project. Preloading dateutil before the main bruteFit imports has been the most
# reliable way to keep batch mode from touching those hooks during pandas/matplotlib
# startup.
from dateutil import rrule, tz  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bruteFit.batch import batch_fit_run, batch_fit_tree, discover_run_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch process and fit bruteFit run folders."
    )
    parser.add_argument(
        "path",
        help="Dataset root, or one run folder when used with --single-run.",
    )
    parser.add_argument("--top-n", type=int, default=3, help="Number of best fits to export.")
    parser.add_argument(
        "--metric",
        default="redchi",
        choices=("redchi", "bic", "residual_rms", "combo"),
        help="Metric used to rank exported fits.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Worker count. macOS is still forced to 1 by the fitter for stability.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run folder names to include, for example --runs Q0 Q1 B Q0_Q1.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List discovered run folders without fitting.",
    )
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Treat path as one run folder instead of a dataset root.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    path = Path(args.path).expanduser().resolve()
    run_names = set(args.runs) if args.runs else None

    if args.single_run:
        run_dirs = [path]
    else:
        run_dirs = discover_run_dirs(path, run_names=run_names)

    if args.dry_run:
        for run_dir in run_dirs:
            print(run_dir)
        print(f"Discovered {len(run_dirs)} run folder(s).")
        return 0

    if args.single_run:
        manifest = batch_fit_run(
            path,
            top_n=args.top_n,
            metric=args.metric,
            processes=args.processes,
        )
        print(f"Wrote batch outputs to {manifest['output_dir']}")
        return 0

    manifests = batch_fit_tree(
        path,
        top_n=args.top_n,
        metric=args.metric,
        processes=args.processes,
        run_names=run_names,
    )
    print(f"Completed {len(manifests)} run folder(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
