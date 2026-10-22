"""Export a label sheet (CSV) for one lecture: the concepts C2 would score, ready for raters.

Runs the difficulty pipeline with its default setup (no network, no semantic scorer) on a PPTX or
PDF and writes one row per concept, with the concept ids exactly as C2 emits them. The rater
columns (difficulty_label, rater_id, student_level, confidence, notes, labelled_at) are left empty.
The sheet holds concept ids, labels and slide numbers only, never slide text.

Without --graph the slide-title mock graph is used; pass a C1 graph JSON to use C1's concepts.
Give each lecture a pseudonymous --lecture-id such as L001; do not use a course or lecturer name.

Exit codes: 0 ok, 2 bad arguments, unreadable or rejected input, no concepts, or an existing file.
"""

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.ids import slug_id
from edusync_c2.difficulty.materials import MaterialsError
from edusync_c2.difficulty.schemas import C1GraphResponse

COLUMNS = (
    "lecture_id",
    "concept_id",
    "concept_slug",
    "concept_label",
    "slide_indexes",
    "difficulty_label",
    "rater_id",
    "student_level",
    "confidence",
    "notes",
    "labelled_at",
    "rubric_version",
    "synthetic",
)
LECTURE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")


def check_lecture_id(value: str) -> str:
    """Return ``value`` if it is a pseudonymous id (letters, digits, - and _, no slashes)."""
    if not LECTURE_ID_PATTERN.match(value):
        raise argparse.ArgumentTypeError(
            f"{value!r} is not a valid lecture id: use 1-32 letters, digits, '-' or '_' "
            "(for example L001), no spaces or path separators"
        )
    return value


def build_rows(lecture_id: str, rubric_version: str, predictions: list) -> list[list[str]]:
    """Sheet rows in a deterministic order: first slide, then concept id."""
    ordered = sorted(
        predictions,
        key=lambda p: (min(p.slide_indexes) if p.slide_indexes else 10**9, p.concept_id),
    )
    rows = []
    for p in ordered:
        slides = ";".join(str(i) for i in sorted(set(p.slide_indexes)))
        rows.append(
            [
                lecture_id,
                p.concept_id,
                slug_id(p.label),
                p.label,
                slides,
                "",
                "",
                "",
                "",
                "",
                "",
                rubric_version,
                "false",
            ]
        )
    return rows


def sheet_text(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return buf.getvalue()


def _emit(text: str) -> None:
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


def main(argv: list[str] | None = None) -> int:
    """Write the sheet; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", required=True, type=Path, help="Lecture file (.pptx or .pdf)")
    parser.add_argument("--lecture-id", required=True, type=check_lecture_id)
    parser.add_argument("--graph", type=Path, help="C1 graph JSON (default: the slide-title mock)")
    parser.add_argument("--rubric-version", default="r1")
    parser.add_argument(
        "--out", type=Path, help="Output CSV (default: <lecture-id>_label_sheet.csv)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return int(exc.code or 0)
    if not args.rubric_version.strip() or any(c in args.rubric_version for c in ",\r\n"):
        print("--rubric-version must be a non-empty single token without commas", file=sys.stderr)
        return 2

    out = args.out or Path(f"{args.lecture_id}_label_sheet.csv")
    if out.exists() and not args.force:
        print(f"{out} already exists; pass --force to overwrite it", file=sys.stderr)
        return 2

    try:
        data = args.file.read_bytes()
    except OSError as exc:
        print(f"Cannot read {args.file}: {exc}", file=sys.stderr)
        return 2
    graph = None
    if args.graph:
        try:
            graph = C1GraphResponse.model_validate(
                json.loads(args.graph.read_text(encoding="utf-8-sig"))
            )
        except (OSError, ValueError, ValidationError) as exc:
            print(f"Cannot use graph {args.graph}: {type(exc).__name__}", file=sys.stderr)
            return 2

    try:
        response = DifficultyPipeline().run(data, args.file.name, graph)
    except MaterialsError as exc:
        print(f"Lecture file rejected: {exc.message}", file=sys.stderr)
        return 2

    rows = build_rows(args.lecture_id, args.rubric_version, response.predictions)
    if not rows:
        print("No concepts found in this lecture; pass a C1 graph with --graph", file=sys.stderr)
        return 2
    try:
        with open(out, "w", encoding="utf-8", newline="") as handle:
            handle.write(sheet_text(rows))
    except OSError as exc:
        print(f"Cannot write {out}: {exc}", file=sys.stderr)
        return 2
    _emit(f"Wrote {out}: {len(rows)} concept(s) for lecture {args.lecture_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
