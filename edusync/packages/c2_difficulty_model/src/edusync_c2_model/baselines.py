"""Honesty-gate baselines evaluated on one split: rule model, majority class, feature-only logistic.

Every runner returns a ``BaselineResult``: predicted class ids (``None`` = the model abstained, a
coverage miss) and a metrics dict from ``metrics.evaluate`` computed on the covered rows, plus the
coverage. ``compare`` renders results as a markdown table; ``human_ceiling`` gives the
human-human QWK from the rater columns of a label CSV.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edusync_c2.difficulty.models import RuleBaselineModel
from edusync_c2.difficulty.registry import FEATURE_NAMES
from edusync_c2.difficulty.schemas import FeatureStatus, FusionFeature, FusionInput
from edusync_c2_model.examples import Example
from edusync_c2_model.labels import ADJUDICATOR, LABEL_ID, LABEL_ORDER, read_rows
from edusync_c2_model.metrics import confusion_matrix, evaluate, metrics_from_confusion
from edusync_c2_model.models.experts import to_arrays
from edusync_c2_model.models.scaling import Standardizer
from edusync_c2_model.models.softmax import SoftmaxRegression


@dataclass(frozen=True)
class BaselineResult:
    name: str
    predictions: list[int | None]  # class id per example, None = no prediction
    metrics: dict  # ``metrics.evaluate`` on the covered rows, plus "coverage" and "n_total"


def fusion_input_from_example(example: Example) -> FusionInput:
    """The ``FusionInput`` the rule model would have seen for this stored example."""
    features = [
        FusionFeature(
            name=name,
            value=None if missing else value,
            status=FeatureStatus(status),
            missing=bool(missing),
        )
        for name, value, missing, status in zip(
            FEATURE_NAMES, example.values, example.missing, example.statuses, strict=True
        )
    ]
    return FusionInput(concept_id=example.concept_id, text=example.text, features=features)


def score_predictions(
    name: str, examples: Sequence[Example], predictions: Sequence[int | None], *, n_boot: int = 200
) -> BaselineResult:
    """Metrics on the examples that have a prediction; coverage = share that do."""
    covered = [(e, p) for e, p in zip(examples, predictions, strict=True) if p is not None]
    metrics = evaluate(
        [e.label_id for e, _ in covered],
        [int(p) for _, p in covered],
        [e.lecture_id for e, _ in covered],
        n_boot=n_boot,
        synthetic=any(e.synthetic for e in examples),
    )
    metrics["n_total"] = len(examples)
    metrics["coverage"] = len(covered) / len(examples) if examples else None
    return BaselineResult(name, list(predictions), metrics)


def run_rule_baseline(examples: Sequence[Example], *, n_boot: int = 200) -> BaselineResult:
    """``RuleBaselineModel`` over the stored features; ``None`` outputs are coverage misses."""
    outputs = RuleBaselineModel().predict([fusion_input_from_example(e) for e in examples])
    predictions = [None if o is None else LABEL_ID[o.label.value] for o in outputs]
    return score_predictions("rule baseline", examples, predictions, n_boot=n_boot)


def majority_class(train: Sequence[Example]) -> int:
    """Most frequent training class id (the lowest id wins ties)."""
    if not train:
        raise ValueError("cannot take a majority class from an empty training set")
    return int(np.argmax(np.bincount([e.label_id for e in train], minlength=len(LABEL_ORDER))))


def run_majority_baseline(
    train: Sequence[Example], test: Sequence[Example], *, n_boot: int = 200
) -> BaselineResult:
    return score_predictions(
        "majority class", test, [majority_class(train)] * len(test), n_boot=n_boot
    )


def _feature_design(
    arrays_values: np.ndarray, arrays_missing: np.ndarray, scaler: Standardizer
) -> np.ndarray:
    return np.hstack([scaler.transform(arrays_values, arrays_missing), arrays_missing * 1.0])


def run_logistic_baseline(
    train: Sequence[Example],
    test: Sequence[Example],
    *,
    l2: float = 0.01,
    seed: int = 0,
    n_boot: int = 200,
) -> BaselineResult:
    """Feature-only multinomial logistic regression (train-only scaling, mask columns)."""
    if not train:
        raise ValueError("cannot fit a logistic baseline on an empty training set")
    fit_arrays = to_arrays(train)
    scaler = Standardizer.fit(fit_arrays.values, fit_arrays.missing)
    head = SoftmaxRegression(l2=l2).fit(
        _feature_design(fit_arrays.values, fit_arrays.missing, scaler), fit_arrays.y, seed
    )
    test_arrays = to_arrays(test)
    probs = head.predict_proba(_feature_design(test_arrays.values, test_arrays.missing, scaler))
    predictions: list[int | None] = [int(p) for p in np.argmax(probs, axis=1)]
    return score_predictions("feature logistic", test, predictions, n_boot=n_boot)


def human_ceiling(label_csv: Path) -> dict | None:
    """QWK between the first two raters (by rater id) per concept, or ``None`` without pairs."""
    by_concept: dict[tuple[str, str], dict[str, int]] = {}
    for row in read_rows(label_csv):
        rater, label = row["rater_id"], row["difficulty_label"]
        if rater == ADJUDICATOR or label not in LABEL_ID:
            continue
        by_concept.setdefault((row["lecture_id"], row["concept_id"]), {})[rater] = LABEL_ID[label]
    pairs = [
        (ids[sorted(ids)[0]], ids[sorted(ids)[1]])
        for _key, ids in sorted(by_concept.items())
        if len(ids) >= 2
    ]
    if not pairs:
        return None
    first, second = [p[0] for p in pairs], [p[1] for p in pairs]
    stats = metrics_from_confusion(confusion_matrix(first, second))
    return {"qwk": stats["qwk"], "mae": stats["mae"], "n_pairs": len(pairs)}


def compare(results: Sequence[BaselineResult], ceiling: dict | None = None) -> str:
    """A markdown table of the metrics of every result (and the human ceiling when given)."""
    columns = ("qwk", "mae", "adjacent_accuracy", "balanced_accuracy")
    header = ["model", "n", "coverage", "QWK", "MAE", "adjacent acc", "balanced acc"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        cells = [r.name, str(r.metrics["n"]), _num(r.metrics["coverage"])]
        cells += [_num(r.metrics[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    if ceiling is not None:
        cells = ["human-human (raters 1 and 2)", str(ceiling["n_pairs"]), "n/a"]
        cells += [_num(ceiling["qwk"]), _num(ceiling["mae"]), "n/a", "n/a"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
