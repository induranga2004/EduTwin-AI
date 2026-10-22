"""Generate the SYNTHETIC per-model datasets (A language, B prerequisites, C explanation, D density,
M combiner) and optionally extract their features through the frozen C2 pipeline.

SYNTHETIC DATA: plumbing only, never accuracy. Output goes to a git-ignored folder (default
packages/c2_difficulty_model/data/synthetic); lecture files, manifests and label CSVs are never
committed. With --extract the features are cached (default packages/c2_difficulty_model/cache), so
a second run is fast, and a class-balance table is printed.

Exit codes: 0 ok, 2 bad arguments, existing different dataset without --force, or extraction error.
"""

import argparse
import sys
import time
from pathlib import Path

from edusync_c2_model.extract import ExtractError
from edusync_c2_model.synthetic import (
    DEFAULT_BASE_SEED,
    DEFAULT_DATA_DIR,
    DEFAULT_ROWS,
    SyntheticError,
    build_expert_datasets,
    class_balance,
    load_dataset,
)

DEFAULT_CACHE_DIR = DEFAULT_DATA_DIR.parents[1] / "cache" / "synthetic"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR, help="Output folder")
    parser.add_argument("--rows-per-expert", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes for deck building")
    parser.add_argument("--force", action="store_true", help="Replace a dataset that differs")
    parser.add_argument("--extract", action="store_true", help="Extract features and join labels")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR, help="Feature cache")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return int(exc.code or 0)

    started = time.perf_counter()
    try:
        datasets = build_expert_datasets(
            args.out, args.rows_per_expert, args.seed, jobs=args.jobs, force=args.force
        )
    except SyntheticError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"SYNTHETIC DATA written to {args.out} in {time.perf_counter() - started:.1f}s")
    for letter, paths in datasets.items():
        state = "reused" if paths.reused else "written"
        print(f"  {letter} {paths.name:<14} {paths.rows} concepts, {paths.decks} decks ({state})")
    if not args.extract:
        return 0

    print(f"{'dataset':<14} {'rows':>6} {'Low':>7} {'Medium':>7} {'High':>7}  cache  unmatched")
    for paths in datasets.values():
        began = time.perf_counter()
        try:
            examples, joined, stats = load_dataset(paths, args.cache / paths.name)
        except ExtractError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        share = class_balance(examples)
        print(
            f"{paths.name:<14} {len(examples):>6} {share['Low']:>7.1%} {share['Medium']:>7.1%} "
            f"{share['High']:>7.1%}  {stats.hits}h/{stats.misses}m  "
            f"{len(joined.unmatched_labels)}+{len(joined.unlabelled)}  "
            f"({time.perf_counter() - began:.1f}s)"
        )
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
