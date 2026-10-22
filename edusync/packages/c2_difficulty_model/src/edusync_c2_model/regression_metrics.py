"""Regression and rank metrics with seeded bootstrap-by-group confidence intervals (numpy only).

Anything undefined (fewer than 2 rows, a constant series, zero variance in R2) is ``None``, never
NaN. Bootstrap replicates resample whole groups (passages, articles, lectures) with replacement.
"""

from collections.abc import Callable, Sequence

import numpy as np


def _pair(y_true: Sequence[float], y_pred: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-D and of the same length")
    return a, b


def rank_average(values: Sequence[float]) -> np.ndarray:
    """1-based ranks; tied values share the mean of the ranks they occupy."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start
        while end + 1 < array.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def pearson(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    a, b = _pair(y_true, y_pred)
    if a.size < 2:
        return None
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a**2).sum() * (b**2).sum()))
    if den == 0.0:
        return None
    return float((a * b).sum() / den)


def spearman(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    a, b = _pair(y_true, y_pred)
    if a.size < 2:
        return None
    return pearson(rank_average(a), rank_average(b))


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    a, b = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else None


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else None


def r2(y_true: Sequence[float], y_pred: Sequence[float]) -> float | None:
    a, b = _pair(y_true, y_pred)
    total = float(((a - a.mean()) ** 2).sum()) if a.size else 0.0
    if total == 0.0:
        return None
    return 1.0 - float(((a - b) ** 2).sum()) / total


def bootstrap_by_group(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[str],
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float | None]],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float] | None]:
    """Percentile CI per metric from resampling whole groups; undefined replicates are skipped."""
    a, b = _pair(y_true, y_pred)
    if len(groups) != a.size:
        raise ValueError("groups must have the same length as y_true")
    names = sorted(set(groups))
    members: list[list[int]] = [[] for _ in names]
    index = {g: i for i, g in enumerate(names)}
    for row, group in enumerate(groups):
        members[index[group]].append(row)
    arrays = [np.asarray(m, dtype=np.int64) for m in members]
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_boot if names else 0):
        picks = rng.integers(0, len(names), size=len(names))
        rows = np.concatenate([arrays[p] for p in picks])
        for name, fn in metrics.items():
            value = fn(a[rows], b[rows])
            if value is not None:
                samples[name].append(value)
    out: dict[str, dict[str, float] | None] = {}
    for name, values in samples.items():
        if not values:
            out[name] = None
            continue
        low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
        out[name] = {"low": float(low), "high": float(high)}
    return out


def evaluate_regressor(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    groups: Sequence[str],
    seed: int = 0,
    *,
    n_boot: int = 1000,
) -> dict:
    """All point metrics, sample sizes and bootstrap-by-group CIs for Spearman and RMSE."""
    return {
        "n": len(y_true),
        "n_groups": len(set(groups)),
        "spearman": spearman(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "ci": bootstrap_by_group(
            y_true, y_pred, groups, {"spearman": spearman, "rmse": rmse}, n_boot=n_boot, seed=seed
        ),
        "n_boot": n_boot,
        "seed": seed,
    }
