"""Classification metrics (numpy only)."""

from typing import Any

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Rows are true classes, columns are predicted classes."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred), strict=True):
        cm[int(t), int(p)] += 1
    return cm


def _prf(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.diag(cm).astype(float)
    pred = cm.sum(axis=0).astype(float)
    true = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    recall = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    _, _, f1 = _prf(confusion_matrix(y_true, y_pred, n_classes))
    return float(f1.mean())


def classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, f1 and support."""
    cm = confusion_matrix(y_true, y_pred, len(labels))
    precision, recall, f1 = _prf(cm)
    support = cm.sum(axis=1)
    return {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }


def summarize(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Accuracy, macro-F1, per-class report and confusion matrix in one JSON-able dict."""
    n = len(labels)
    return {
        "num_examples": int(np.asarray(y_true).size),
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred, n),
        "per_class": classification_report(y_true, y_pred, labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, n).tolist(),
        "labels": list(labels),
    }


def selection_score(
    summary: dict[str, Any], select_by: str, positive_label: str | None
) -> tuple[float, float]:
    """Sortable epoch score (higher is better): the chosen metric first, macro-F1 as tie-break."""
    if select_by == "macro_f1":
        return (summary["macro_f1"], 0.0)
    if select_by == "positive_recall":
        if positive_label not in summary["per_class"]:
            raise ValueError(f"positive_label {positive_label!r} not in {list(summary['per_class'])}")
        return (summary["per_class"][positive_label]["recall"], summary["macro_f1"])
    raise ValueError(f"Unknown select_by {select_by!r}")
