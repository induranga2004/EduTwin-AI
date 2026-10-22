"""The four group experts: small logistic heads over a frozen embedding and/or a feature group.

A: embedding of the concept text + group A (language) features + missing mask
B: group B (prerequisite) features + mask (no text)
C: embedding + group C (explanation) features + mask
D: group D (density) features + mask (no text)

An expert abstains ("no vote") when it has nothing to look at: every feature of its group is
missing and, for a text expert, there is no text either. A text expert with text never abstains.
During training features are randomly masked (seeded) so the model stays usable when a group is
missing at serving time (no C1 graph, semantic scorer off, PDF without structure).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from edusync_c2.difficulty.registry import FEATURE_NAMES, features_of
from edusync_c2.difficulty.schemas import FeatureGroup
from edusync_c2_model.examples import Example
from edusync_c2_model.labels import LABEL_ORDER
from edusync_c2_model.models.scaling import Standardizer
from edusync_c2_model.models.softmax import SoftmaxRegression

N_CLASSES = len(LABEL_ORDER)


@dataclass(frozen=True)
class ExpertSpec:
    letter: str
    group: FeatureGroup
    use_text: bool

    @property
    def name(self) -> str:
        return self.group.value


EXPERT_SPECS: tuple[ExpertSpec, ...] = (
    ExpertSpec("A", FeatureGroup.LANGUAGE, True),
    ExpertSpec("B", FeatureGroup.PREREQUISITES, False),
    ExpertSpec("C", FeatureGroup.EXPLANATION, True),
    ExpertSpec("D", FeatureGroup.DENSITY, False),
)


@dataclass(frozen=True)
class ExampleArrays:
    """The numeric view of a list of examples (all features, registry order)."""

    values: np.ndarray  # (n, F) float64, 0.0 where missing
    missing: np.ndarray  # (n, F) bool
    has_text: np.ndarray  # (n,) bool
    y: np.ndarray  # (n,) int64 class ids

    def __len__(self) -> int:
        return len(self.y)

    def take(self, index: np.ndarray) -> "ExampleArrays":
        return ExampleArrays(
            self.values[index], self.missing[index], self.has_text[index], self.y[index]
        )


def to_arrays(examples: Sequence[Example]) -> ExampleArrays:
    n_features = len(FEATURE_NAMES)
    return ExampleArrays(
        values=np.array([e.values for e in examples], dtype=np.float64).reshape(-1, n_features),
        missing=np.array([e.missing for e in examples], dtype=bool).reshape(-1, n_features),
        has_text=np.array([bool(e.text.strip()) for e in examples], dtype=bool),
        y=np.array([e.label_id for e in examples], dtype=np.int64),
    )


def canonical_order(examples: Sequence[Example]) -> list[Example]:
    """Examples sorted by (lecture, concept, text): training never depends on input order."""
    return sorted(examples, key=lambda e: (e.lecture_id, e.concept_id, e.text))


def mask_dropout(
    missing: np.ndarray, rate: float, rng: np.random.Generator, *, whole_rows: bool
) -> np.ndarray:
    """A copy of ``missing`` with random extra cells masked (and, if asked, whole rows)."""
    drop = rng.random(missing.shape) < rate
    if whole_rows:
        drop = drop | (rng.random(len(missing)) < rate)[:, None]
    return missing | drop


class GroupExpert:
    """One expert: standardiser + logistic head, trained on masked and unmasked rows."""

    def __init__(
        self,
        spec: ExpertSpec,
        *,
        l2: float = 0.01,
        ordinal: bool = False,
        mask_dropout_rate: float = 0.2,
        class_weight: str | Sequence[float] | None = "balanced",
        max_iter: int = 300,
    ) -> None:
        self.spec = spec
        self.mask_dropout_rate = mask_dropout_rate
        self.columns = np.array([FEATURE_NAMES.index(n) for n in features_of(spec.group)])
        self.feature_names = features_of(spec.group)
        self.head = SoftmaxRegression(
            N_CLASSES, l2=l2, ordinal=ordinal, class_weight=class_weight, max_iter=max_iter
        )
        self.standardizer: Standardizer | None = None
        self.n_embedding = 0
        self.active = False  # False until trained on at least one row the expert can vote on

    # ------------------------------------------------------------------ inputs

    def votes(self, arrays: ExampleArrays, group_missing: np.ndarray | None = None) -> np.ndarray:
        """True where the expert has something to look at."""
        miss = arrays.missing[:, self.columns] if group_missing is None else group_missing
        seen = (~miss).any(axis=1)
        return (seen | (arrays.has_text & self.spec.use_text)) & self.active

    def _raw(
        self, arrays: ExampleArrays, group_missing: np.ndarray, embeddings: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = arrays.values[:, self.columns]
        if not self.spec.use_text:
            return values, group_missing
        if embeddings is None or len(embeddings) != len(arrays):
            raise ValueError(f"expert {self.spec.letter} needs one embedding per example")
        return (
            np.hstack([embeddings, values]),
            np.hstack([np.zeros(embeddings.shape, dtype=bool), group_missing]),
        )

    def _design(
        self, arrays: ExampleArrays, group_missing: np.ndarray, embeddings: np.ndarray | None
    ) -> np.ndarray:
        assert self.standardizer is not None
        raw, raw_missing = self._raw(arrays, group_missing, embeddings)
        return np.hstack([self.standardizer.transform(raw, raw_missing), group_missing * 1.0])

    # ---------------------------------------------------------------- training

    def fit(
        self, arrays: ExampleArrays, embeddings: np.ndarray | None, seed: int = 0
    ) -> "GroupExpert":
        group_missing = arrays.missing[:, self.columns]
        self.active = True
        keep = self.votes(arrays)
        raw, raw_missing = self._raw(arrays, group_missing, embeddings)
        self.n_embedding = raw.shape[1] - len(self.columns)
        if not keep.any():  # nothing to learn from (e.g. no C1 graph): abstain on everything
            self.active = False
            self.standardizer = Standardizer(np.zeros(raw.shape[1]), np.ones(raw.shape[1]))
            self.head.fit_uniform(raw.shape[1] + len(self.columns))
            return self
        self.standardizer = Standardizer.fit(raw[keep], raw_missing[keep])
        designs = [self._design(arrays.take(keep), group_missing[keep], _rows(embeddings, keep))]
        labels = [arrays.y[keep]]
        if self.mask_dropout_rate > 0:
            rng = np.random.default_rng(seed)
            dropped = mask_dropout(
                group_missing, self.mask_dropout_rate, rng, whole_rows=self.spec.use_text
            )
            keep_aug = self.votes(arrays, dropped)
            if keep_aug.any():
                designs.append(
                    self._design(
                        arrays.take(keep_aug), dropped[keep_aug], _rows(embeddings, keep_aug)
                    )
                )
                labels.append(arrays.y[keep_aug])
        self.head.fit(np.vstack(designs), np.concatenate(labels), seed)
        return self

    # -------------------------------------------------------------- prediction

    def predict_proba(
        self, arrays: ExampleArrays, embeddings: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(probabilities (n, K), votes (n,))``; non-voting rows get the head's prior."""
        design = self._design(arrays, arrays.missing[:, self.columns], embeddings)
        return self.head.predict_proba(design), self.votes(arrays)

    def feature_z(self, arrays: ExampleArrays) -> np.ndarray:
        """Standardised group features ``(n, g)``, exactly 0 where missing."""
        assert self.standardizer is not None
        sl = slice(self.n_embedding, None)
        return Standardizer(self.standardizer.mean[sl], self.standardizer.std[sl]).transform(
            arrays.values[:, self.columns], arrays.missing[:, self.columns]
        )

    def contributions(self, arrays: ExampleArrays, class_ids: np.ndarray) -> np.ndarray:
        """``(n, g)``: standardised value times the (class-centred) weight for ``class_ids``."""
        coef = self.head.class_coefficients()
        g = len(self.columns)
        weights = coef[self.n_embedding : self.n_embedding + g][:, class_ids].T
        return self.feature_z(arrays) * weights

    # ------------------------------------------------------------ serialisation

    def to_arrays(self) -> dict[str, np.ndarray]:
        assert self.standardizer is not None
        prefix = f"expert.{self.spec.letter}"
        return {
            **self.standardizer.to_arrays(prefix),
            **self.head.to_arrays(prefix),
            f"{prefix}.active": np.array(int(self.active)),
        }

    @classmethod
    def from_arrays(
        cls, spec: ExpertSpec, data: dict[str, np.ndarray], *, ordinal: bool
    ) -> "GroupExpert":
        prefix = f"expert.{spec.letter}"
        expert = cls(spec, ordinal=ordinal)
        expert.standardizer = Standardizer.from_arrays(data, prefix)
        expert.head = SoftmaxRegression.from_arrays(data, prefix, ordinal=ordinal)
        expert.n_embedding = len(expert.standardizer.mean) - len(expert.columns)
        expert.active = bool(data[f"{prefix}.active"])
        return expert


def _rows(embeddings: np.ndarray | None, index: np.ndarray) -> np.ndarray | None:
    return None if embeddings is None else embeddings[index]
