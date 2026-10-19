"""Baseline coverage classifier training (needs the ``research`` extra).

Usage: ``python -m edusync_c1.research.train_model --dataset coverage_examples.csv``

This is an initial, unverified baseline: interpret results cautiously on small datasets.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from edusync_c1.research.dataset import (
    GROUP_COLUMN,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TEXT_COLUMN,
    Row,
    load_dataset,
)
from edusync_core.settings import get_settings

MIN_EXAMPLES = 20
MIN_LECTURES = 3
RANDOM_STATE = 42


def check_trainable(rows: list[Row]) -> None:
    """Raise ValueError if validated rows are too few / too uniform for a held-out evaluation."""
    if len(rows) < MIN_EXAMPLES:
        raise ValueError(
            f"Only {len(rows)} examples found. Add at least {MIN_EXAMPLES} labelled "
            "examples before attempting this initial training run."
        )
    lectures = {r[GROUP_COLUMN] for r in rows}
    if len(lectures) < MIN_LECTURES:
        raise ValueError(
            f"Found {len(lectures)} distinct lecture IDs. Use at least {MIN_LECTURES} "
            "distinct lectures for the initial held-out evaluation."
        )
    if len({r[TARGET_COLUMN] for r in rows}) < 2:
        raise ValueError("Training requires at least two different labels.")


def train(
    dataset_path: Path, model_dir: Path, report_dir: Path
) -> dict[str, Any]:  # pragma: no cover
    """Train on held-out-by-lecture split; save model and metrics, return the metrics."""
    import joblib
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rows = load_dataset(dataset_path)
    check_trainable(rows)
    data = pd.DataFrame(rows)
    features = data[[TEXT_COLUMN, *NUMERIC_COLUMNS]].copy()
    features[TEXT_COLUMN] = features[TEXT_COLUMN].astype(str)
    y = data[TARGET_COLUMN].astype(str)
    groups = data[GROUP_COLUMN].astype(str)

    # Keep each lecture entirely in either training or test data.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(features, y, groups=groups))
    train_lectures, test_lectures = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
    if train_lectures & test_lectures:
        raise RuntimeError("Data leakage detected: a lecture is in both train and test sets.")
    if y.iloc[train_idx].nunique() < 2:
        raise ValueError(
            "The training split contains fewer than two labels. "
            "Add more varied examples across lectures."
        )

    transform = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True),
                TEXT_COLUMN,
            ),
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                list(NUMERIC_COLUMNS),
            ),
        ]
    )
    model = Pipeline(
        [
            ("features", transform),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    model.fit(features.iloc[train_idx], y.iloc[train_idx])
    y_test = y.iloc[test_idx]
    predictions = model.predict(features.iloc[test_idx])
    labels = sorted(y.unique())

    metrics = {
        "model_name": "TF-IDF and numerical-feature Logistic Regression",
        "total_examples": len(data),
        "training_examples": len(train_idx),
        "test_examples": len(test_idx),
        "training_lectures": sorted(train_lectures),
        "test_lectures": sorted(test_lectures),
        "labels_in_dataset": labels,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(),
        "confusion_matrix_labels": labels,
        "random_state": RANDOM_STATE,
        "evaluation_note": (
            "Initial held-out lecture evaluation. "
            "Interpret results cautiously when the dataset is small."
        ),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "coverage_classifier.joblib")
    (report_dir / "coverage_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; paths default to ``<EDUSYNC_DATA_DIR>/c1/...``."""
    base = get_settings().data_dir / "c1"
    parser = argparse.ArgumentParser(description="Train the baseline coverage classifier.")
    parser.add_argument(
        "--dataset", type=Path, default=base / "annotations" / "coverage_examples.csv"
    )
    parser.add_argument("--model-dir", type=Path, default=base / "models")
    parser.add_argument("--report-dir", type=Path, default=base / "reports")
    args = parser.parse_args(argv)

    try:
        metrics = train(args.dataset, args.model_dir, args.report_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"Accuracy: {metrics['accuracy']:.3f}  Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Model and metrics saved under {args.model_dir} and {args.report_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
