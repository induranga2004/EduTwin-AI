"""Ingest the real lecture decks: draft graphs, extract label-free features, write a summary.

Per deck, ``graphs/<stem>.draft.json`` is drafted when missing and ``graphs/<stem>.graph.json`` (the
reviewed copy used for extraction, marked curated but unreviewed) is created from it when missing;
existing graph files are never overwritten. Features go through the frozen pipeline with the
default concept filter, no network and no semantic scorer (semantic features are unavailable),
and are cached. Writes ``<decks>/../manifest.csv`` and a counts-only markdown summary.
"""

import argparse
import sys
from pathlib import Path

from edusync_c2_model.ingest import ingest_decks, render_summary

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REAL = PACKAGE_ROOT / "data" / "real"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decks", type=Path, default=REAL / "decks")
    parser.add_argument("--graphs", type=Path, default=REAL / "graphs")
    parser.add_argument("--course", default="ISSA")
    parser.add_argument("--cache", type=Path, default=PACKAGE_ROOT / "cache" / "real")
    parser.add_argument("--manifest", type=Path, help="default: <decks>/../manifest.csv")
    parser.add_argument(
        "--summary", type=Path, default=PACKAGE_ROOT / "runs" / "real_ingest_summary.md"
    )
    parser.add_argument(
        "--refresh-auto",
        action="store_true",
        help="regenerate the draft and rewrite graph.json only where review_status is "
        "'auto-draft, unreviewed' (edited graphs are left alone and reported)",
    )
    args = parser.parse_args(argv)
    manifest = args.manifest or args.decks.parent / "manifest.csv"
    try:
        results = ingest_decks(
            args.decks,
            args.graphs,
            args.cache,
            manifest,
            course=args.course,
            refresh_auto=args.refresh_auto,
        )
    except OSError as exc:
        print(f"error: {type(exc).__name__}", file=sys.stderr)
        return 1
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(render_summary(results, course=args.course), encoding="utf-8")
    failed = [r for r in results if r.error]
    print(f"{len(results) - len(failed)} of {len(results)} decks ingested; summary: {args.summary}")
    if args.refresh_auto:
        refreshed = sum(r.refreshed for r in results)
        skipped = [r.lecture_id for r in results if r.refresh_skipped]
        print(f"refresh-auto: {refreshed} refreshed, {len(skipped)} left alone (edited)")
        for lecture_id in skipped:
            print(f"  {lecture_id}: graph was edited, not refreshed")
    for r in failed:
        print(f"  {r.lecture_id}: {r.error}", file=sys.stderr)
    return 1 if failed or not results else 0


if __name__ == "__main__":
    sys.exit(main())
