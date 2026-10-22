"""Combine the five signals (A, B, C, D, text) into one score, from the reference's structure.

Weights: loadings of the first principal component of the signal correlation matrix over the
reference concepts, sign-aligned to be positive, shrunk 50% toward equal weights and normalised.
A concept uses only the signals it has (weights renormalised). The final score is the percentile
of the combined raw score among the reference concepts. Cut-offs are percentiles of that score:
top 25% High, bottom 40% Low. Unvalidated defaults, not tuned on any lecturer rating.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edusync_c2.difficulty.scoring.reference import ScoringError, percentile_in, quantile_table

SIGNALS = ("A", "B", "C", "D", "text")
DEFAULT_LOW = 0.40
DEFAULT_HIGH = 0.75
SHRINK = 0.5
MIN_ROWS = 3
COMBINER_FORMAT = 1
COMBINER_NAME = "combiner.json"

SignalScores = Mapping[str, float | None]


def label_for(score: float | None, low: float, high: float) -> str | None:
    if score is None:
        return None
    if score >= high:
        return "High"
    return "Low" if score < low else "Medium"


@dataclass(frozen=True)
class Combiner:
    weights: dict[str, float]
    low: float
    high: float
    combined_table: list[float]

    def raw(self, signals: SignalScores) -> float | None:
        used = {k: v for k, v in signals.items() if v is not None and self.weights.get(k, 0.0) > 0}
        total = sum(self.weights[k] for k in used)
        if not used:
            return None
        return sum(self.weights[k] * v for k, v in used.items()) / total

    def contributions(self, signals: SignalScores) -> dict[str, float | None]:
        """Share of the raw score from each signal (sums to the raw score); None if unused."""
        used = {k: v for k, v in signals.items() if v is not None and self.weights.get(k, 0.0) > 0}
        total = sum(self.weights[k] for k in used)
        return {k: (self.weights[k] * used[k] / total if k in used else None) for k in SIGNALS}

    def score(self, signals: SignalScores) -> float | None:
        raw = self.raw(signals)
        return None if raw is None else percentile_in(self.combined_table, raw)

    def label(self, score: float | None) -> str | None:
        return label_for(score, self.low, self.high)

    def to_dict(self) -> dict:
        return {
            "format": COMBINER_FORMAT,
            "signals": list(SIGNALS),
            "weights": self.weights,
            "cutoffs": {"low": self.low, "high": self.high},
            "combined_table": self.combined_table,
            "method": "first principal component, sign-aligned, shrunk 50% toward equal weights",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Combiner":
        try:
            if data["format"] != COMBINER_FORMAT:
                raise ScoringError("unsupported combiner format")
            weights = {k: float(data["weights"][k]) for k in SIGNALS}
            low, high = float(data["cutoffs"]["low"]), float(data["cutoffs"]["high"])
            table = [float(x) for x in data["combined_table"]]
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ScoringError(f"invalid combiner: {type(exc).__name__}") from exc
        if not table or not 0 <= low <= high <= 1 or min(weights.values()) < 0:
            raise ScoringError("invalid combiner values")
        return cls(weights, low, high, table)

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: Path) -> "Combiner":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise ScoringError(f"cannot read the combiner: {type(exc).__name__}") from exc


def fit_weights(matrix: np.ndarray) -> dict[str, float]:
    """Weights per signal from an ``(n, 5)`` score matrix with NaN for unavailable entries."""
    n_rows = matrix.shape[0]
    valid = ~np.isnan(matrix)
    usable = [
        j
        for j in range(len(SIGNALS))
        if valid[:, j].sum() >= MIN_ROWS and np.nanstd(matrix[:, j]) > 1e-9
    ]
    if not usable:
        usable = [j for j in range(len(SIGNALS)) if valid[:, j].any()]
    equal = np.zeros(len(SIGNALS))
    equal[usable] = 1.0 / len(usable) if usable else 0.0
    if len(usable) < 2 or n_rows < MIN_ROWS:
        return {k: float(equal[j]) for j, k in enumerate(SIGNALS)}
    cols = matrix[:, usable]
    means = np.nanmean(cols, axis=0)
    cols = np.where(np.isnan(cols), means, cols)
    std = cols.std(axis=0)
    z = (cols - means) / np.where(std < 1e-12, 1.0, std)
    corr = np.nan_to_num(np.corrcoef(z, rowvar=False))
    _, vectors = np.linalg.eigh(corr)
    loading = vectors[:, -1]
    if loading.sum() < 0:
        loading = -loading
    loading = np.clip(loading, 0.0, None)  # a unit vector with sum >= 0 keeps a positive entry
    pc = np.zeros(len(SIGNALS))
    pc[usable] = loading / loading.sum()
    weights = SHRINK * equal + (1 - SHRINK) * pc
    weights = weights / weights.sum()
    return {k: float(weights[j]) for j, k in enumerate(SIGNALS)}


def fit_combiner(
    rows: Sequence[SignalScores], *, low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH
) -> Combiner:
    """Fit weights and the combined-score distribution from reference signal scores."""
    matrix = np.array(
        [[np.nan if r.get(k) is None else float(r[k]) for k in SIGNALS] for r in rows],
        dtype=np.float64,
    ).reshape(len(rows), len(SIGNALS))
    weights = fit_weights(matrix)
    shell = Combiner(weights, low, high, [0.0])
    raws = [v for v in (shell.raw(r) for r in rows) if v is not None]
    table = quantile_table(raws)
    if not table:
        raise ScoringError("no reference concept has any signal")
    return Combiner(weights, low, high, table)
