"""Ordinal classification metrics for Low/Medium/High (numpy and plain Python only).

Class ids are Low=0, Medium=1, High=2. Anything that is undefined (no rows, a single class in both
raters, a class that never occurs) is ``None``, never 0 or NaN.
"""

from collections.abc import Sequence

import numpy as np

from edusync_c2_model.labels import LABEL_ORDER

N_CLASSES = len(LABEL_ORDER)
METRIC_NAMES = ("qwk", "mae", "accuracy", "adjacent_accuracy", "balanced_accuracy")


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> np.ndarray:
    """``cm[i, j]`` = rows with true class ``i`` predicted as ``j``."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if not (0 <= t < N_CLASSES and 0 <= p < N_CLASSES):
            raise ValueError(f"class ids must be 0..{N_CLASSES - 1}, got {t} and {p}")
        cm[t, p] += 1
    return cm


def _qwk(cm: np.ndarray) -> float | None:
    n = cm.sum()
    if n == 0:
        return None
    observed = cm / n
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    idx = np.arange(N_CLASSES)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (N_CLASSES - 1) ** 2
    den = float((weights * expected).sum())
    if den == 0:
        return None
    return 1.0 - float((weights * observed).sum()) / den


def _per_class(cm: np.ndarray) -> list[dict]:
    out = []
    for i, name in enumerate(LABEL_ORDER):
        tp, support, predicted = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else None
        )
        out.append(
            {"label": name, "precision": precision, "recall": recall, "f1": f1, "support": support}
        )
    return out


def metrics_from_confusion(cm: np.ndarray) -> dict:
    """All scalar metrics plus the per-class report for one confusion matrix."""
    n = int(cm.sum())
    per_class = _per_class(cm)
    recalls = [c["recall"] for c in per_class if c["recall"] is not None]
    idx = np.arange(N_CLASSES)
    distance = np.abs(idx[:, None] - idx[None, :])
    return {
        "n": n,
        "qwk": _qwk(cm),
        "mae": float((cm * distance).sum()) / n if n else None,
        "accuracy": float(np.trace(cm)) / n if n else None,
        "adjacent_accuracy": float(cm[distance <= 1].sum()) / n if n else None,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def bootstrap_ci(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    lecture_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float] | None]:
    """Percentile CI per metric from resampling whole lectures with replacement.

    A metric is ``None`` when it is undefined in every replicate; undefined replicates are skipped.
    """
    if not len(y_true) == len(y_pred) == len(lecture_ids):
        raise ValueError("y_true, y_pred and lecture_ids must have the same length")
    groups = sorted(set(lecture_ids))
    per_lecture = np.zeros((len(groups), N_CLASSES, N_CLASSES), dtype=np.int64)
    index = {g: i for i, g in enumerate(groups)}
    for t, p, g in zip(y_true, y_pred, lecture_ids, strict=True):
        per_lecture[index[g], t, p] += 1
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    for _ in range(n_boot if groups else 0):
        picks = rng.integers(0, len(groups), size=len(groups))
        stats = metrics_from_confusion(per_lecture[picks].sum(axis=0))
        for name in METRIC_NAMES:
            if stats[name] is not None:
                samples[name].append(stats[name])
    out: dict[str, dict[str, float] | None] = {}
    for name, values in samples.items():
        if not values:
            out[name] = None
            continue
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        out[name] = {"low": float(low), "high": float(high)}
    return out


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    lecture_ids: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    synthetic: bool = False,
) -> dict:
    """Metrics, bootstrap CIs and sample sizes in one JSON-friendly dict."""
    result = metrics_from_confusion(confusion_matrix(y_true, y_pred))
    result["n_lectures"] = len(set(lecture_ids))
    result["ci"] = bootstrap_ci(y_true, y_pred, lecture_ids, n_boot=n_boot, seed=seed)
    result["n_boot"] = n_boot
    result["synthetic"] = synthetic
    return result
