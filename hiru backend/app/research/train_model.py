
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "coverage_examples.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "coverage_classifier.joblib"
METRICS_PATH = REPORT_DIR / "coverage_model_metrics.json"

TEXT_COLUMN = "transcript_text"

NUMERIC_COLUMNS = [
    "semantic_similarity",
    "prerequisite_coverage",
    "previous_evidence_count",
]

TARGET_COLUMN = "label"
GROUP_COLUMN = "lecture_id"

MIN_EXAMPLES = 20
MIN_LECTURES = 3


def main():
    print("=" * 60)
    print("EduTwin-AI: Coverage Classifier Training")
    print("=" * 60)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    data = pd.read_csv(DATASET_PATH)

    required = [
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

    missing = [
        column for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing dataset columns: {missing}"
        )

    if data.empty:
        raise ValueError(
            "The CSV contains no labelled examples. "
            "Add real, manually labelled examples first."
        )

    if data[required].isnull().any().any():
        raise ValueError(
            "The dataset contains empty required values."
        )

    if len(data) < MIN_EXAMPLES:
        raise ValueError(
            f"Only {len(data)} examples found. "
            f"Add at least {MIN_EXAMPLES} labelled examples "
            "before attempting this initial training run."
        )

    lecture_count = data[GROUP_COLUMN].nunique()

    if lecture_count < MIN_LECTURES:
        raise ValueError(
            f"Found {lecture_count} distinct lecture IDs. "
            f"Use at least {MIN_LECTURES} distinct lectures "
            "for the initial held-out evaluation."
        )

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise",
        )

    for column in [
        "semantic_similarity",
        "prerequisite_coverage",
    ]:
        if not data[column].between(0, 1).all():
            raise ValueError(
                f"{column} must be between 0 and 1."
            )

    if (data["previous_evidence_count"] < 0).any():
        raise ValueError(
            "previous_evidence_count cannot be negative."
        )

    if (data["timestamp_seconds"] < 0).any():
        raise ValueError(
            "timestamp_seconds cannot be negative."
        )

    if data[TARGET_COLUMN].nunique() < 2:
        raise ValueError(
            "Training requires at least two different labels."
        )

    print(f"Examples: {len(data)}")
    print(f"Distinct lectures: {lecture_count}")
    print("\nLabel distribution:")
    print(data[TARGET_COLUMN].value_counts().to_string())

    X = data[
        [TEXT_COLUMN] + NUMERIC_COLUMNS
    ].copy()

    X[TEXT_COLUMN] = X[TEXT_COLUMN].astype(str)

    y = data[TARGET_COLUMN].astype(str)
    groups = data[GROUP_COLUMN].astype(str)

    # Keep each lecture entirely in either training or test data.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    train_lectures = set(groups.iloc[train_indices])
    test_lectures = set(groups.iloc[test_indices])

    if train_lectures.intersection(test_lectures):
        raise RuntimeError(
            "Data leakage detected: a lecture appears "
            "in both training and test sets."
        )

    if y_train.nunique() < 2:
        raise ValueError(
            "The training split contains fewer than two labels. "
            "Add more varied examples across lectures."
        )

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Test examples: {len(X_test)}")
    print(f"Training lectures: {len(train_lectures)}")
    print(f"Test lectures: {len(test_lectures)}")

    text_pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    features = ColumnTransformer(
        transformers=[
            (
                "text",
                text_pipeline,
                TEXT_COLUMN,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_COLUMNS,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("\nTraining the classifier...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    labels = sorted(y.unique())

    metrics = {
        "model_name": "TF-IDF and numerical-feature Logistic Regression",
        "dataset_path": str(DATASET_PATH),
        "total_examples": int(len(data)),
        "training_examples": int(len(X_train)),
        "test_examples": int(len(X_test)),
        "training_lectures": sorted(train_lectures),
        "test_lectures": sorted(test_lectures),
        "labels_in_dataset": labels,
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        ).tolist(),
        "confusion_matrix_labels": labels,
        "random_state": 42,
        "evaluation_note": (
            "Initial held-out lecture evaluation. "
            "Interpret results cautiously when the dataset is small."
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    print("\nEvaluation results:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print("\nTraining and evaluation completed.")


if __name__ == "__main__":
    main()
