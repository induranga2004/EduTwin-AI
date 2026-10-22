"""Stacking combiner: a softmax head over the experts' probabilities and their vote masks.

Inputs per example are ``E x K`` expert probabilities (zeroed for experts that abstained) plus
``E`` vote flags. Training randomly switches experts off (seeded vote dropout), so the head learns
to rely on whichever experts are present. A row where nobody votes gets the head's class prior.
"""

import numpy as np

from edusync_c2_model.models.softmax import SoftmaxRegression


def combiner_features(probs: np.ndarray, votes: np.ndarray) -> np.ndarray:
    """``(n, E*K + E)``: masked probabilities followed by the vote flags."""
    probs = np.asarray(probs, dtype=np.float64)
    votes = np.asarray(votes, dtype=bool)
    masked = probs * votes[:, :, None]
    flat = masked.reshape(len(probs), probs.shape[1] * probs.shape[2])
    return np.hstack([flat, votes * 1.0])


class StackingCombiner:
    def __init__(
        self,
        *,
        n_classes: int = 3,
        l2: float = 0.01,
        vote_dropout: float = 0.25,
        class_weight: str | None = "balanced",
        max_iter: int = 300,
    ) -> None:
        self.vote_dropout = vote_dropout
        self.head = SoftmaxRegression(
            n_classes, l2=l2, class_weight=class_weight, max_iter=max_iter
        )

    def fit(
        self, probs: np.ndarray, votes: np.ndarray, y: np.ndarray, seed: int = 0
    ) -> "StackingCombiner":
        """Fit on (out-of-fold) expert probabilities; ``votes`` is ``(n, E)`` bool."""
        votes = np.asarray(votes, dtype=bool)
        y = np.asarray(y, dtype=np.int64)
        keep = votes.any(axis=1)
        if not keep.any():
            raise ValueError("no row has a vote from any expert")
        parts = [(probs[keep], votes[keep], y[keep])]
        if self.vote_dropout > 0:
            rng = np.random.default_rng(seed)
            thinned = votes & (rng.random(votes.shape) >= self.vote_dropout)
            keep_aug = thinned.any(axis=1)
            parts.append((probs[keep_aug], thinned[keep_aug], y[keep_aug]))
        X = np.vstack([combiner_features(p, v) for p, v, _ in parts])
        self.head.fit(X, np.concatenate([part[2] for part in parts]), seed)
        return self

    def predict_proba(self, probs: np.ndarray, votes: np.ndarray) -> np.ndarray:
        return self.head.predict_proba(combiner_features(probs, votes))

    def to_arrays(self) -> dict[str, np.ndarray]:
        return self.head.to_arrays("combiner")

    @classmethod
    def from_arrays(cls, data: dict[str, np.ndarray]) -> "StackingCombiner":
        combiner = cls()
        combiner.head = SoftmaxRegression.from_arrays(data, "combiner", ordinal=False)
        return combiner
