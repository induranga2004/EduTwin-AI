"""The group ensemble: four experts plus a stacking combiner, trained leak-free.

Two training schemes, both with no lecture ever scored by a model that trained on it:

* ``GroupEnsemble.fit``: one labelled set. Out-of-fold expert probabilities from a grouped k-fold
  over the training lectures feed the combiner; the experts used at serving time are refitted on
  all training lectures.
* ``GroupEnsemble.fit_from_example_sets`` / ``fit_experts_from_datasets``: one dataset per expert
  plus a combiner dataset. Each expert is fitted on its own dataset; the combiner is fitted on the
  experts' probabilities for the combiner dataset's training lectures. Lecture ids are asserted
  disjoint across all datasets, so those probabilities are genuinely out-of-sample and no k-fold
  refit is needed.

Everything is deterministic: examples are put in a canonical order before fitting, seeds are
explicit and float64 numpy does the rest.
"""

import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from edusync_c2.difficulty.registry import FEATURE_SCHEMA_VERSION, feature_schema_hash
from edusync_c2_model.examples import Example, read_examples
from edusync_c2_model.labels import LABEL_ORDER
from edusync_c2_model.models.combiner import StackingCombiner
from edusync_c2_model.models.embeddings import Embedder
from edusync_c2_model.models.experts import (
    EXPERT_SPECS,
    ExampleArrays,
    ExpertSpec,
    GroupExpert,
    canonical_order,
    to_arrays,
)
from edusync_c2_model.split import SplitError, check_disjoint, group_kfold

FORMAT_VERSION = 1
N_CLASSES = len(LABEL_ORDER)


@dataclass(frozen=True)
class EnsembleParams:
    l2: float = 0.01
    ordinal: bool = False
    mask_dropout: float = 0.2
    combiner_l2: float = 0.01
    vote_dropout: float = 0.25
    class_weight: str | None = "balanced"
    max_iter: int = 300

    def make_expert(self, spec: ExpertSpec) -> GroupExpert:
        return GroupExpert(
            spec,
            l2=self.l2,
            ordinal=self.ordinal,
            mask_dropout_rate=self.mask_dropout,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )

    def make_combiner(self) -> StackingCombiner:
        return StackingCombiner(
            l2=self.combiner_l2,
            vote_dropout=self.vote_dropout,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
        )


@dataclass(frozen=True)
class EnsembleOutput:
    probs: np.ndarray  # (n, K) combined probabilities
    expert_probs: np.ndarray  # (n, E, K)
    votes: np.ndarray  # (n, E) bool; False = the expert abstained


def _rows(embeddings: np.ndarray | None, index: np.ndarray) -> np.ndarray | None:
    return None if embeddings is None else embeddings[index]


def _embed_texts(embed: Embedder | None, examples: Sequence[Example]) -> np.ndarray:
    if embed is None:
        raise ValueError("an embed callable is required (experts A and C read the concept text)")
    return np.asarray(embed([e.text for e in examples]), dtype=np.float64)


def _expert_outputs(
    experts: Sequence[GroupExpert], arrays: ExampleArrays, embeddings: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    probs = np.zeros((len(arrays), len(experts), N_CLASSES))
    votes = np.zeros((len(arrays), len(experts)), dtype=bool)
    for i, expert in enumerate(experts):
        probs[:, i], votes[:, i] = expert.predict_proba(arrays, embeddings)
    return probs, votes


class GroupEnsemble:
    def __init__(
        self,
        experts: Sequence[GroupExpert],
        combiner: StackingCombiner,
        params: EnsembleParams,
        *,
        embed: Embedder | None = None,
        encoder_id: str = "",
    ) -> None:
        self.experts = tuple(experts)
        self.combiner = combiner
        self.params = params
        self.embed = embed
        self.encoder_id = encoder_id
        self.fold_lectures: list[tuple[list[str], list[str]]] = []  # (fit, predict) per OOF fold
        self.oof_probs: np.ndarray | None = None
        self.oof_votes: np.ndarray | None = None

    # ---------------------------------------------------------------- training

    @classmethod
    def fit(
        cls,
        train_examples: Sequence[Example],
        seed: int = 0,
        folds: int = 5,
        *,
        embed: Embedder | None = None,
        encoder_id: str = "",
        params: EnsembleParams | None = None,
    ) -> "GroupEnsemble":
        """Fit on one labelled set; the combiner sees only out-of-fold expert probabilities."""
        params = params or EnsembleParams()
        examples = canonical_order(train_examples)
        arrays = to_arrays(examples)
        embeddings = _embed_texts(embed, examples)
        lectures = np.array([e.lecture_id for e in examples])
        oof = np.zeros((len(examples), len(EXPERT_SPECS), N_CLASSES))
        oof_votes = np.zeros((len(examples), len(EXPERT_SPECS)), dtype=bool)
        used = []
        for fit_lectures, predict_lectures in group_kfold(set(lectures.tolist()), folds, seed):
            fit_idx = np.flatnonzero(np.isin(lectures, fit_lectures))
            predict_idx = np.flatnonzero(np.isin(lectures, predict_lectures))
            check_disjoint(lectures[fit_idx].tolist(), lectures[predict_idx].tolist())
            used.append((fit_lectures, predict_lectures))
            fold_experts = [
                params.make_expert(s).fit(arrays.take(fit_idx), _rows(embeddings, fit_idx), seed)
                for s in EXPERT_SPECS
            ]
            p, v = _expert_outputs(
                fold_experts, arrays.take(predict_idx), _rows(embeddings, predict_idx)
            )
            oof[predict_idx], oof_votes[predict_idx] = p, v
        experts = [params.make_expert(s).fit(arrays, embeddings, seed) for s in EXPERT_SPECS]
        combiner = params.make_combiner().fit(oof, oof_votes, arrays.y, seed)
        ensemble = cls(experts, combiner, params, embed=embed, encoder_id=encoder_id)
        ensemble.fold_lectures, ensemble.oof_probs, ensemble.oof_votes = used, oof, oof_votes
        return ensemble

    @classmethod
    def fit_from_example_sets(
        cls,
        expert_examples: Mapping[str, Sequence[Example]],
        combiner_examples: Sequence[Example],
        seed: int = 0,
        *,
        embed: Embedder | None = None,
        encoder_id: str = "",
        params: EnsembleParams | None = None,
    ) -> "GroupEnsemble":
        """Each expert on its own examples (keys A..D), the combiner on a lecture-disjoint set."""
        params = params or EnsembleParams()
        letters = [s.letter for s in EXPERT_SPECS]
        if sorted(expert_examples) != letters:
            raise ValueError(f"expert_examples needs exactly the keys {letters}")
        sets = {k: canonical_order(v) for k, v in expert_examples.items()}
        sets["combiner"] = canonical_order(combiner_examples)
        names = sorted(sets)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                check_disjoint(
                    {e.lecture_id for e in sets[first]}, {e.lecture_id for e in sets[second]}
                )
        experts = []
        for spec in EXPERT_SPECS:
            own = sets[spec.letter]
            emb = _embed_texts(embed, own) if spec.use_text else None
            experts.append(params.make_expert(spec).fit(to_arrays(own), emb, seed))
        ensemble = cls(experts, params.make_combiner(), params, embed=embed, encoder_id=encoder_id)
        arrays = to_arrays(sets["combiner"])
        probs, votes = _expert_outputs(experts, arrays, _embed_texts(embed, sets["combiner"]))
        ensemble.combiner.fit(probs, votes, arrays.y, seed)
        ensemble.oof_probs, ensemble.oof_votes = probs, votes
        return ensemble

    # -------------------------------------------------------------- prediction

    def predict_detail(self, examples: Sequence[Example]) -> EnsembleOutput:
        if not examples:
            empty = np.zeros((0, len(self.experts), N_CLASSES))
            return EnsembleOutput(empty[:, 0], empty, np.zeros((0, len(self.experts)), dtype=bool))
        arrays = to_arrays(examples)
        probs, votes = _expert_outputs(self.experts, arrays, _embed_texts(self.embed, examples))
        return EnsembleOutput(self.combiner.predict_proba(probs, votes), probs, votes)

    def predict_proba(self, examples: Sequence[Example]) -> np.ndarray:
        return self.predict_detail(examples).probs

    def predict(self, examples: Sequence[Example]) -> np.ndarray:
        return np.argmax(self.predict_proba(examples), axis=1)

    # ----------------------------------------------------------- serialisation

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Arrays only (``np.savez`` ready, loadable without pickle)."""
        config = {
            "format": FORMAT_VERSION,
            "labels": list(LABEL_ORDER),
            "experts": [s.letter for s in EXPERT_SPECS],
            "params": asdict(self.params),
            "encoder_id": self.encoder_id,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_schema_hash": feature_schema_hash(),
        }
        data: dict[str, np.ndarray] = {"config": np.array(json.dumps(config, sort_keys=True))}
        for expert in self.experts:
            data.update(expert.to_arrays())
        data.update(self.combiner.to_arrays())
        return data

    @classmethod
    def from_npz_dict(
        cls, data: Mapping[str, np.ndarray], *, embed: Embedder | None = None
    ) -> "GroupEnsemble":
        try:
            config = json.loads(str(data["config"]))
            params = EnsembleParams(**config["params"])
            arrays = {k: np.asarray(v) for k, v in data.items() if k != "config"}
            experts = [
                GroupExpert.from_arrays(s, arrays, ordinal=params.ordinal) for s in EXPERT_SPECS
            ]
            combiner = StackingCombiner.from_arrays(arrays)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"not a valid ensemble artifact: {exc}") from exc
        if config.get("format") != FORMAT_VERSION or config.get("labels") != list(LABEL_ORDER):
            raise ValueError("unsupported ensemble format or label order")
        if (
            config.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or config.get("feature_schema_hash") != feature_schema_hash()
        ):
            raise ValueError("the ensemble was trained on a different feature schema")
        return cls(experts, combiner, params, embed=embed, encoder_id=config.get("encoder_id", ""))


def fit_experts_from_datasets(
    paths_per_expert: Mapping[str, Path],
    combiner_path: Path,
    seed: int = 0,
    *,
    embed: Embedder | None = None,
    encoder_id: str = "",
    params: EnsembleParams | None = None,
) -> GroupEnsemble:
    """Like ``fit_from_example_sets`` but reading each dataset from an examples JSONL file."""
    return GroupEnsemble.fit_from_example_sets(
        {k: read_examples(Path(p)) for k, p in paths_per_expert.items()},
        read_examples(Path(combiner_path)),
        seed,
        embed=embed,
        encoder_id=encoder_id,
        params=params,
    )


def split_by_lecture(
    examples: Sequence[Example],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> tuple[list[Example], list[Example], list[Example]]:
    """Train/validation/test examples with whole lectures in exactly one part (seeded)."""
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise SplitError("fractions must be positive and sum to less than 1")
    lectures = sorted({e.lecture_id for e in examples})
    if len(lectures) < 3:
        raise SplitError(f"need at least 3 lectures to split three ways, got {len(lectures)}")
    random.Random(seed).shuffle(lectures)
    n_val = max(1, round(val_fraction * len(lectures)))
    n_test = max(1, round(test_fraction * len(lectures)))
    val, test = set(lectures[:n_val]), set(lectures[n_val : n_val + n_test])
    parts: tuple[list[Example], list[Example], list[Example]] = ([], [], [])
    for e in canonical_order(examples):
        parts[1 if e.lecture_id in val else 2 if e.lecture_id in test else 0].append(e)
    return parts
