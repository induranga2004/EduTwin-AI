"""Train-only target scaling and Low/Medium/High cut-offs for the public text corpora.

Everything is fitted on the TRAIN part only and stored in a plain JSON dict with the fit
statistics, so val/test never influence a statistic. Direction: higher = harder. CLEAR ease is
converted to difficulty by negating it before the z-score.
"""

import math
from collections.abc import Sequence

import numpy as np

TARGETS_VERSION = 1
MIN_STD = 1e-12


class TargetError(ValueError):
    """The target statistics cannot be fitted or applied."""


def _array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise TargetError("need at least 2 finite training targets")
    return array


def fit_difficulty_scaler(train_targets: Sequence[float], *, higher_is_easier: bool) -> dict:
    """Mean/std of the training difficulty (``-target`` when higher means easier)."""
    sign = -1.0 if higher_is_easier else 1.0
    difficulty = sign * _array(train_targets)
    std = float(difficulty.std())
    return {
        "version": TARGETS_VERSION,
        "sign": sign,
        "mean": float(difficulty.mean()),
        "std": std if std > MIN_STD else 1.0,
        "n_train": int(difficulty.size),
    }


def to_difficulty(targets: Sequence[float], scaler: dict) -> np.ndarray:
    """z-scored difficulty, higher = harder, using the training mean/std."""
    raw = scaler["sign"] * np.asarray(targets, dtype=np.float64)
    return (raw - scaler["mean"]) / scaler["std"]


def fit_cutoffs(
    train_difficulty: Sequence[float], *, low_share: float = 1 / 3, high_share: float = 1 / 3
) -> dict:
    """Cut-offs so ``low_share`` of TRAIN is Low and ``high_share`` is High (default terciles)."""
    if not (0 < low_share < 1 and 0 < high_share < 1 and low_share + high_share < 1):
        raise TargetError("low_share and high_share must be in (0, 1) and sum to less than 1")
    values = _array(train_difficulty)
    low, high = np.quantile(values, [low_share, 1 - high_share])
    return {
        "version": TARGETS_VERSION,
        "low_share": low_share,
        "high_share": high_share,
        "low_below": float(low),
        "high_from": float(high),
        "n_train": int(values.size),
    }


def to_label_ids(difficulty: Sequence[float], cutoffs: dict) -> list[int]:
    """Low=0 (< low_below), High=2 (>= high_from), otherwise Medium=1."""
    out = []
    for value in difficulty:
        if not math.isfinite(value):
            raise TargetError("difficulty must be finite")
        if value < cutoffs["low_below"]:
            out.append(0)
        elif value >= cutoffs["high_from"]:
            out.append(2)
        else:
            out.append(1)
    return out


def fit_targets(train_targets: Sequence[float], *, higher_is_easier: bool) -> dict:
    """Scaler plus tercile cut-offs on the scaled training difficulty, as one JSON-ready dict."""
    scaler = fit_difficulty_scaler(train_targets, higher_is_easier=higher_is_easier)
    cutoffs = fit_cutoffs(to_difficulty(train_targets, scaler))
    return {"scaler": scaler, "cutoffs": cutoffs}
