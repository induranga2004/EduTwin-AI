"""Reference distributions: per-feature quantile tables over a set of reference lectures.

A percentile says where a value sits among the reference concepts (0 = lowest, 1 = highest). Ties
take the mid-rank. The tables are plain sorted lists of quantiles, so a percentile is a bisect.
"""

import json
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from edusync_c2.difficulty.registry import FEATURE_NAMES

N_QUANTILES = 101
PROVISIONAL_BELOW = 30  # fewer reference decks than this: the reference is provisional
REFERENCE_FORMAT = 1


class ScoringError(ValueError):
    """A scoring artifact is missing, malformed or incompatible."""


class RowLike(Protocol):
    """What ``build_reference`` needs from a feature row (``FeatureRow`` satisfies it)."""

    lecture_id: str
    values: Sequence[float | None]
    missing: Sequence[int]
    statuses: Sequence[str]


def quantile_table(values: Iterable[float], n: int = N_QUANTILES) -> list[float]:
    """``n`` evenly spaced quantiles of ``values`` (empty list for no values)."""
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return []
    return [float(v) for v in np.quantile(data, np.linspace(0.0, 1.0, n))]


def percentile_in(table: Sequence[float], value: float) -> float | None:
    """Position of ``value`` in a sorted quantile table, in [0, 1]; None for an empty table.

    Below the minimum 0, above the maximum 1, between two quantiles linear, equal to a run of tied
    quantiles the middle of the run.
    """
    n = len(table)
    if n == 0:
        return None
    if n == 1:
        return 0.5
    low, high = bisect_left(table, value), bisect_right(table, value)
    if low < high:  # value equals table[low:high]
        return ((low + high - 1) / 2) / (n - 1)
    if low == 0:
        return 0.0
    if low == n:
        return 1.0
    lo_v, hi_v = table[low - 1], table[low]
    return (low - 1 + (value - lo_v) / (hi_v - lo_v)) / (n - 1)


@dataclass(frozen=True)
class Reference:
    """Quantile tables per feature plus how many lectures and concepts built them."""

    tables: dict[str, list[float]]
    counts: dict[str, int]
    n_decks: int
    n_concepts: int
    feature_schema_version: str = ""
    feature_schema_hash: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def provisional(self) -> bool:
        return self.n_decks < PROVISIONAL_BELOW

    def percentile(self, name: str, value: float) -> float | None:
        return percentile_in(self.tables.get(name, []), value)

    def to_dict(self) -> dict:
        return {
            "format": REFERENCE_FORMAT,
            "n_decks": self.n_decks,
            "n_concepts": self.n_concepts,
            "provisional": self.provisional,
            "feature_schema_version": self.feature_schema_version,
            "feature_schema_hash": self.feature_schema_hash,
            "counts": self.counts,
            "tables": self.tables,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Reference":
        try:
            if data["format"] != REFERENCE_FORMAT:
                raise ScoringError("unsupported reference format")
            tables = {str(k): [float(x) for x in v] for k, v in data["tables"].items()}
            counts = {str(k): int(v) for k, v in data["counts"].items()}
            return cls(
                tables=tables,
                counts=counts,
                n_decks=int(data["n_decks"]),
                n_concepts=int(data["n_concepts"]),
                feature_schema_version=str(data.get("feature_schema_version", "")),
                feature_schema_hash=str(data.get("feature_schema_hash", "")),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ScoringError(f"invalid reference: {type(exc).__name__}") from exc

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    @classmethod
    def load(cls, path: Path) -> "Reference":
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:  # JSON and format errors are ValueErrors
            raise ScoringError(f"cannot read the reference: {type(exc).__name__}") from exc


def usable(row: RowLike, index: int) -> bool:
    """Was feature ``index`` really computed (not missing, not mocked, not unavailable)?"""
    return (
        row.statuses[index] == "computed"
        and not row.missing[index]
        and row.values[index] is not None
    )


def build_reference(
    rows: Sequence[RowLike], *, schema_version: str = "", schema_hash: str = ""
) -> Reference:
    """Per-feature quantile tables of the computed values over all rows."""
    tables: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for i, name in enumerate(FEATURE_NAMES):
        values = [float(r.values[i]) for r in rows if usable(r, i)]
        tables[name] = quantile_table(values)
        counts[name] = len(values)
    return Reference(
        tables=tables,
        counts=counts,
        n_decks=len({r.lecture_id for r in rows}),
        n_concepts=len(rows),
        feature_schema_version=schema_version,
        feature_schema_hash=schema_hash,
    )
