"""Evaluate the cheap text baselines on CLEAR and OneStopEnglish (whichever files are present).

Splits are seeded by passage/article and sealed in ``<data>/splits/<source>_split.json`` (written
once, then only read). The markdown report goes to the git-ignored ``runs/`` folder. These are
general-English corpora, not technical lectures.
"""

import argparse
import sys
from pathlib import Path

from edusync_c2_model.baselines_text import render_baseline_report, run_baselines
from edusync_c2_model.public_data.clear import load_clear
from edusync_c2_model.public_data.licences import SOURCES
from edusync_c2_model.public_data.onestop import load_onestop
from edusync_c2_model.public_data.splits import (
    make_group_split,
    partition,
    read_split,
    write_split_once,
)
from edusync_c2_model.split import SplitError

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAVEAT = (
    "Caveat: general English with human readability judgements, not technical lectures. "
    "These numbers say nothing about accuracy on lecture slides."
)
LOADERS = {"clear": (load_clear, True), "onestop": (load_onestop, False)}


def evaluate_source(
    key: str, data_dir: Path, runs_dir: Path, *, seed: int, n_boot: int
) -> str | None:
    """Report text for ``key``, or None when its data file is absent; also writes the report."""
    source = next(s for s in SOURCES if s.key == key)
    path = data_dir / source.filename
    if not path.is_file():
        print(f"{key}: {path} not found, skipped")
        return None
    loader, higher_is_easier = LOADERS[key]
    examples = loader(path)
    split_path = data_dir / "splits" / f"{key}_split.json"
    if split_path.exists():
        split = read_split(split_path)
    else:
        split = make_group_split({e.group_id for e in examples}, seed=seed, source=key)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        write_split_once(split_path, split)
    outcome = run_baselines(
        partition(examples, split), higher_is_easier=higher_is_easier, seed=seed, n_boot=n_boot
    )
    report = render_baseline_report(key, outcome, CAVEAT)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"baselines_text_{key}.md").write_text(report, encoding="utf-8", newline="\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE_ROOT / "data" / "public")
    parser.add_argument("--runs-dir", type=Path, default=PACKAGE_ROOT / "runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args(argv)
    try:
        reports = [
            evaluate_source(key, args.data_dir, args.runs_dir, seed=args.seed, n_boot=args.n_boot)
            for key in LOADERS
        ]
    except (SplitError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for report in filter(None, reports):
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
