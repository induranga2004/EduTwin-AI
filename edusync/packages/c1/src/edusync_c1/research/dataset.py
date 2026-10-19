"""Human-labelled coverage examples: the single label schema and validation (stdlib only)."""

import csv
from pathlib import Path
from typing import Any

COVERAGE_LABELS = frozenset(
    {"not_yet_observed", "introduced", "adequately_covered", "insufficiently_covered"}
)

REQUIRED_COLUMNS = (
    "lecture_id",
    "segment_id",
    "timestamp_seconds",
    "concept",
    "transcript_text",
    "semantic_similarity",
    "prerequisite_coverage",
    "previous_evidence_count",
    "label",
)
TEXT_COLUMN = "transcript_text"
NUMERIC_COLUMNS = ("semantic_similarity", "prerequisite_coverage", "previous_evidence_count")
UNIT_INTERVAL_COLUMNS = ("semantic_similarity", "prerequisite_coverage")
TARGET_COLUMN = "label"
GROUP_COLUMN = "lecture_id"

Row = dict[str, Any]


def _to_float(row: Row, column: str) -> float:
    try:
        return float(row[column])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{column} must be numeric, got {row[column]!r}.") from exc


def validate_rows(rows: list[Row]) -> list[Row]:
    """Validate rows against the label schema; return copies with numeric fields coerced."""
    if not rows:
        raise ValueError("The dataset contains no examples.")
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean: list[Row] = []
    for row in rows:
        if any(row.get(c) is None or str(row[c]).strip() == "" for c in REQUIRED_COLUMNS):
            raise ValueError(
                "Required fields contain empty values. Please complete or correct the annotations."
            )
        out = dict(row)
        for column in ("timestamp_seconds", *NUMERIC_COLUMNS):
            out[column] = _to_float(row, column)
        if out["timestamp_seconds"] < 0:
            raise ValueError("Timestamps cannot be negative.")
        if out["previous_evidence_count"] < 0:
            raise ValueError("Previous evidence counts cannot be negative.")
        for column in UNIT_INTERVAL_COLUMNS:
            if not 0 <= out[column] <= 1:
                raise ValueError(f"{column} values must be between 0 and 1.")
        clean.append(out)

    invalid = {r[TARGET_COLUMN] for r in clean} - COVERAGE_LABELS
    if invalid:
        raise ValueError(f"Invalid coverage labels: {sorted(invalid)}")
    return clean


def load_dataset(csv_path: str | Path) -> list[Row]:
    """Load and validate human-labelled lecture examples."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return validate_rows(rows)


def save_annotation(csv_path: str | Path, example: Row) -> None:
    """Validate and append one labelled example to the annotation CSV."""
    missing = [c for c in REQUIRED_COLUMNS if c not in example]
    if missing:
        raise ValueError(f"Annotation is missing fields: {missing}")
    validate_rows([example])

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (path.exists() and path.stat().st_size > 0)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REQUIRED_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(example)
