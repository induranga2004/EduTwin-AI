
import csv
from pathlib import Path

import pandas as pd


COVERAGE_LABELS = {
    "not_yet_observed",
    "introduced",
    "adequately_covered",
    "insufficiently_covered",
}

REQUIRED_COLUMNS = [
    "lecture_id",
    "segment_id",
    "timestamp_seconds",
    "concept",
    "transcript_text",
    "semantic_similarity",
    "prerequisite_coverage",
    "previous_evidence_count",
    "label",
]


def validate_dataset(data: pd.DataFrame) -> None:
    """Validate the structure and labels of a research dataset."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if data.empty:
        raise ValueError("The dataset contains no examples.")

    if data[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Required fields contain empty values. "
            "Please complete or correct the annotations."
        )

    invalid_labels = set(data["label"]) - COVERAGE_LABELS

    if invalid_labels:
        raise ValueError(
            f"Invalid coverage labels: {sorted(invalid_labels)}"
        )

    numeric_columns = [
        "timestamp_seconds",
        "semantic_similarity",
        "prerequisite_coverage",
        "previous_evidence_count",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    if (data["timestamp_seconds"] < 0).any():
        raise ValueError("Timestamps cannot be negative.")

    if (data["previous_evidence_count"] < 0).any():
        raise ValueError(
            "Previous evidence counts cannot be negative."
        )

    for column in [
        "semantic_similarity",
        "prerequisite_coverage",
    ]:
        if not data[column].between(0, 1).all():
            raise ValueError(
                f"{column} values must be between 0 and 1."
            )


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate human-labelled lecture examples."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    data = pd.read_csv(path)

    validate_dataset(data)

    return data


def save_annotation(
    csv_path: str | Path,
    example: dict,
) -> None:
    """Append one labelled example to the annotation CSV."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in example
    ]

    if missing:
        raise ValueError(
            f"Annotation is missing fields: {missing}"
        )

    row = pd.DataFrame([example], columns=REQUIRED_COLUMNS)
    validate_dataset(row)

    file_exists = path.exists() and path.stat().st_size > 0

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REQUIRED_COLUMNS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(example)
