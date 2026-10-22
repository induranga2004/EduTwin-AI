"""Report how each registered difficulty feature behaves on the seeded synthetic decks.

Builds the synthetic lectures (PPTX and PDF, v1 graphs), runs them through the difficulty
pipeline with the default mock filter (no network) and prints, per feature in registry order
and grouped by group: maturity, status counts (computed / mocked / unavailable) over all concept
predictions, the number of distinct computed values and their min and max, and flags.

Flags: constant (computed with a single distinct value), all-unavailable, planned, partial (some
predictions unavailable). Features flagged constant or all-unavailable that are not planned are
listed in the footer as 'check'.

The synthetic decks make some features constant or unavailable by construction (for example
dens.concepts_per_slide, and every PPTX-structure feature on PDF, which has no structure), so
the flags are hints, not defects.

Exit codes: 0 ok, 2 bad arguments or unwritable output path.
"""

import argparse
import json
import sys

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.registry import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    feature_schema_hash,
)
from edusync_c2.difficulty.schemas import C1GraphResponse, FeatureStatus
from edusync_c2.difficulty.synthetic import (
    build_c1_graph,
    build_pdf,
    build_pptx,
    make_deck_spec,
)

STATUSES = tuple(s.value for s in FeatureStatus)


def collect(seed: int, lectures: int, concepts: int) -> tuple[list[dict], int]:
    """Run the pipeline on every deck; return per-feature stats in registry order and the count."""
    values: dict[str, list[float]] = {s.name: [] for s in FEATURE_SPECS}
    counts: dict[str, dict[str, int]] = {s.name: dict.fromkeys(STATUSES, 0) for s in FEATURE_SPECS}
    predictions = 0
    for k in range(lectures):
        lecture_id = f"synthetic-lecture-{k}"
        spec = make_deck_spec(seed + k, concepts, lecture_id=lecture_id)
        graph = C1GraphResponse.model_validate(build_c1_graph(spec, "v1"))
        for kind, data in (("pptx", build_pptx(spec)), ("pdf", build_pdf(spec))):
            response = DifficultyPipeline().run(data, f"{lecture_id}.{kind}", graph)
            for prediction in response.predictions:
                predictions += 1
                for feature in prediction.features:
                    counts[feature.name][feature.status.value] += 1
                    if feature.status is FeatureStatus.COMPUTED and feature.value is not None:
                        values[feature.name].append(feature.value)

    rows = []
    for spec in FEATURE_SPECS:
        c = counts[spec.name]
        distinct = sorted(set(values[spec.name]))
        flags = []
        if c["computed"] and len(distinct) == 1:
            flags.append("constant")
        if predictions and c["unavailable"] == predictions:
            flags.append("all-unavailable")
        if spec.maturity == "planned":
            flags.append("planned")
        if 0 < c["unavailable"] < predictions:
            flags.append("partial")
        rows.append(
            {
                "name": spec.name,
                "group": spec.group.value,
                "maturity": spec.maturity,
                "computed": c["computed"],
                "mocked": c["mocked"],
                "unavailable": c["unavailable"],
                "distinct": len(distinct),
                "min": distinct[0] if distinct else None,
                "max": distinct[-1] if distinct else None,
                "flags": flags,
            }
        )
    return rows, predictions


def checks(rows: list[dict]) -> list[str]:
    """Names flagged constant or all-unavailable that are not planned."""
    return [
        r["name"]
        for r in rows
        if "planned" not in r["flags"] and ({"constant", "all-unavailable"} & set(r["flags"]))
    ]


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.4g}"


def format_report(rows: list[dict], predictions: int, seed: int, lectures: int) -> str:
    lines = [
        f"Feature report: schema {FEATURE_SCHEMA_VERSION}, {len(rows)} features, "
        f"{predictions} concept predictions (seed {seed}, {lectures} lectures x PPTX+PDF)",
        "Synthetic decks: flags are hints, not defects.",
    ]
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            lines += ["", f"[{group}]"]
        flags = ",".join(r["flags"]) or "-"
        lines.append(
            f"  {r['name']:<34} {r['maturity']:<11} "
            f"computed={r['computed']:<4} mocked={r['mocked']:<4} "
            f"unavailable={r['unavailable']:<4} "
            f"distinct={r['distinct']:<3} min={_num(r['min'])} max={_num(r['max'])}  {flags}"
        )
    lines.append("")
    check = checks(rows)
    if check:
        lines.append("check (constant or all-unavailable, not planned):")
        lines += [f"  check {name}" for name in check]
    else:
        lines.append("check: none")
    return "\n".join(lines)


def snapshot_text() -> str:
    """The registry snapshot, in the format of golden/registry_snapshot.json."""
    payload = {
        "version": FEATURE_SCHEMA_VERSION,
        "hash": feature_schema_hash(),
        "features": [[s.name, s.group.value] for s in FEATURE_SPECS],
    }
    return json.dumps(payload, indent=1) + "\n"


def _emit(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> int:
    """Print the report; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lectures", type=int, default=3)
    parser.add_argument("--concepts", type=int, default=8)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report")
    parser.add_argument("--write-snapshot", metavar="PATH", help="Write the registry snapshot")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return int(exc.code or 0)
    if args.lectures < 1 or args.concepts < 1:
        print("--lectures and --concepts must be at least 1", file=sys.stderr)
        return 2

    if args.write_snapshot:
        try:
            with open(args.write_snapshot, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(snapshot_text())
        except OSError as exc:
            print(f"Cannot write {args.write_snapshot}: {exc}", file=sys.stderr)
            return 2
        return 0

    rows, predictions = collect(args.seed, args.lectures, args.concepts)
    if args.json:
        payload = {
            "version": FEATURE_SCHEMA_VERSION,
            "predictions": predictions,
            "features": rows,
            "check": checks(rows),
        }
        _emit(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _emit(format_report(rows, predictions, args.seed, args.lectures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
