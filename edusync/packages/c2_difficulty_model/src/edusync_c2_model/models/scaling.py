"""Train-only standardisation that understands missing values.

Mean and standard deviation of every column come from the observed (non-missing) training cells
only. A missing cell becomes 0 after standardisation (the training mean), and the caller adds the
missing mask as extra columns so the model can tell "average" from "absent".
"""

import numpy as np

MIN_STD = 1e-8


class Standardizer:
    """Per-column mean/std fitted on training data only."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)

    @classmethod
    def fit(cls, values: np.ndarray, missing: np.ndarray) -> "Standardizer":
        """``values`` and boolean ``missing`` are ``(n, f)``; columns with no data get (0, 1)."""
        values = np.asarray(values, dtype=np.float64)
        observed = ~np.asarray(missing, dtype=bool)
        count = observed.sum(axis=0)
        safe = np.maximum(count, 1)
        mean = np.where(observed, values, 0.0).sum(axis=0) / safe
        centred = np.where(observed, values - mean, 0.0)
        std = np.sqrt((centred**2).sum(axis=0) / safe)
        std = np.where(std < MIN_STD, 1.0, std)
        return cls(np.where(count > 0, mean, 0.0), std)

    def transform(self, values: np.ndarray, missing: np.ndarray) -> np.ndarray:
        """Standardised values with exact zeros in the missing cells."""
        values = np.asarray(values, dtype=np.float64)
        z = (values - self.mean) / self.std
        return np.where(np.asarray(missing, dtype=bool), 0.0, z)

    def to_arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}.mean": self.mean, f"{prefix}.std": self.std}

    @classmethod
    def from_arrays(cls, data: dict[str, np.ndarray], prefix: str) -> "Standardizer":
        return cls(data[f"{prefix}.mean"], data[f"{prefix}.std"])
