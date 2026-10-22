"""Difficulty models: a transparent rule baseline (Phase 1) and a TODO placeholder (Phase 3).

The baseline is NOT trained. It has no probabilities and no confidence; its score is a weighted
mean of a few normalised features, minus a small credit for explanation support. All constants
below are untuned starting points chosen for readability.
"""

from pathlib import Path
from typing import Protocol

from edusync_c2.difficulty.errors import ModelNotAvailableError
from edusync_c2.difficulty.schemas import (
    DifficultyLabel,
    FeatureStatus,
    FusionInput,
    ModelInfo,
    ModelKind,
    ModelOutput,
)


class DifficultyModel(Protocol):
    info: ModelInfo

    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]:
        """One output per input, in order; None = this concept cannot be predicted."""
        ...


# Difficulty-raising components: feature name -> (weight, value at which it saturates at 1.0,
# offset subtracted first). normalised = min(max(x - offset, 0) / span, 1).
# lang.lexical_score and the presence features are already in [0, 1] (span 1, offset 0).
RAISING_WEIGHTS: dict[str, float] = {
    "lang.lexical_score": 0.25,
    "dens.words_per_slide": 0.15,
    "dens.concepts_per_slide": 0.10,
    "dens.formula_presence": 0.10,
    "dens.code_presence": 0.05,
    "prereq.direct_count": 0.10,
    "prereq.chain_depth": 0.10,
}
# name -> (offset, span)
_NORMALISATION: dict[str, tuple[float, float]] = {
    "lang.lexical_score": (0.0, 1.0),
    "dens.words_per_slide": (0.0, 120.0),
    "dens.concepts_per_slide": (1.0, 4.0),
    "dens.formula_presence": (0.0, 1.0),
    "dens.code_presence": (0.0, 1.0),
    "prereq.direct_count": (0.0, 4.0),
    "prereq.chain_depth": (0.0, 4.0),
}
SUPPORT_FEATURES: tuple[str, ...] = (
    "expl.definition_count",
    "expl.example_count",
    "expl.step_count",
)
SUPPORT_CREDIT_MAX = 0.25  # largest amount the support credit can subtract
MIN_COMPONENTS = 3  # fewer computed difficulty-raising components -> no prediction
# Placeholder thresholds, NOT validated. They sit where the baseline scores of the three synthetic
# decks separate (Low up to about 0.11, Medium 0.06-0.24, High above 0.24), so the labels spread
# instead of everything being Low. Matching the synthetic labels is circular (same properties).
# Phase 3 replaces this rule with a model trained on lecturer labels.
LOW_MAX = 0.12  # score < LOW_MAX -> Low
MEDIUM_MAX = 0.25  # score < MEDIUM_MAX -> Medium, otherwise High


def _label(score: float) -> DifficultyLabel:
    if score < LOW_MAX:
        return DifficultyLabel.LOW
    if score < MEDIUM_MAX:
        return DifficultyLabel.MEDIUM
    return DifficultyLabel.HIGH


class RuleBaselineModel:
    """Weighted-mean rule over computed features only. Deterministic, explainable, untrained."""

    info = ModelInfo(
        name="rule_baseline",
        version="0.1",
        kind=ModelKind.BASELINE,
        description=(
            "Rule-based weighted mean of normalised language, density and prerequisite "
            "features minus a small explanation-support credit. Not trained; untuned weights."
        ),
        trained_on=None,
    )

    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]:
        return [self._predict_one(item) for item in inputs]

    @staticmethod
    def _predict_one(item: FusionInput) -> ModelOutput | None:
        computed = {
            f.name: f.value
            for f in item.features
            if f.status is FeatureStatus.COMPUTED and not f.missing and f.value is not None
        }
        used = [n for n in RAISING_WEIGHTS if n in computed]
        if len(used) < MIN_COMPONENTS:
            return None
        explanation: list[str] = []
        weighted = 0.0
        total_weight = 0.0
        for name in used:
            offset, span = _NORMALISATION[name]
            norm = min(max((computed[name] - offset) / span, 0.0), 1.0)
            weight = RAISING_WEIGHTS[name]
            weighted += weight * norm
            total_weight += weight
            explanation.append(f"{name} = {norm:.2f} (computed, weight {weight:.2f})")
        base = weighted / total_weight

        support = [n for n in SUPPORT_FEATURES if n in computed]
        credit = 0.0
        if support:
            credit = SUPPORT_CREDIT_MAX * sum(min(computed[n], 1.0) for n in support) / 3
        score = min(max(base - credit, 0.0), 1.0)
        explanation.append(
            f"support credit = -{credit:.2f} "
            f"({len(support)} of {len(SUPPORT_FEATURES)} support features computed)"
        )
        label = _label(score)
        explanation.append(
            f"score {score:.2f} -> {label.value} (Low < {LOW_MAX}, Medium < {MEDIUM_MAX}, "
            "else High); baseline, not trained"
        )
        return ModelOutput(
            label=label, score=round(score, 4), probabilities=None, explanation=explanation
        )


class MiniLMFusionModel:
    """TODO (Phase 3): MiniLM text encoder plus structured-feature fusion head."""

    info = ModelInfo(
        name="minilm_fusion",
        version="todo",
        kind=ModelKind.TODO,
        description="Not implemented: text encoder fused with the structured features.",
        trained_on=None,
    )

    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]:
        raise NotImplementedError(
            "Phase 3: copy the blueprint into packages/c2_difficulty_model, add the "
            "structured-feature fusion head, train on lecturer labels"
        )


def create_model(name: str = "baseline", model_dir: Path | None = None) -> DifficultyModel:
    """Model factory. Minimal version; the model-interface work adds the artifact-backed model."""
    if name == "baseline":
        return RuleBaselineModel()
    raise ModelNotAvailableError(f"model {name!r} is not available yet")
