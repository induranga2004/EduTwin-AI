"""Difficulty models: a transparent rule baseline (Phase 1) and the trained-model artifact contract.

The baseline is NOT trained. It has no probabilities and no confidence; its score is a weighted
mean of a few normalised features, minus a small credit for explanation support. All constants
below are untuned starting points chosen for readability.

``OnnxFusionModel`` defines and enforces the artifact folder a trained model must ship
(``ARTIFACT_SPEC``). For now it only validates that folder, without importing onnxruntime or
torch; ``predict`` raises ``NotImplementedError`` until inference is implemented.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from edusync_c2.difficulty.errors import ModelArtifactError, ModelNotAvailableError
from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_hash,
)
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
# Regex features only. The expl.semantic_* features are deliberately NOT read here: they exist only
# when the opt-in semantic scorer is on, and the baseline must give the same answer either way.
SUPPORT_FEATURES: tuple[str, ...] = (
    "expl.definition_count",
    "expl.example_count",
    "expl.step_count",
)
SUPPORT_CREDIT_MAX = 0.25  # largest amount the support credit can subtract


def _validate_constants(
    weights: dict[str, float],
    norm: dict[str, tuple[float, float]],
    support: tuple[str, ...],
    names: tuple[str, ...],
) -> None:
    """Fail fast when the baseline constants and the registry drift apart."""
    if set(weights) != set(norm):
        diff = sorted(set(weights) ^ set(norm))
        raise ValueError(f"weights and normalisation keys differ: {', '.join(diff)}")
    unknown = sorted({*weights, *support} - set(names))
    if unknown:
        raise ValueError(f"baseline features not in the registry: {', '.join(unknown)}")
    bad = sorted(name for name, (_offset, span) in norm.items() if not span > 0)
    if bad:
        raise ValueError(f"normalisation span must be > 0 for: {', '.join(bad)}")


_validate_constants(RAISING_WEIGHTS, _NORMALISATION, SUPPORT_FEATURES, FEATURE_NAMES)

MIN_COMPONENTS = 3  # fewer computed difficulty-raising components -> no prediction
# Placeholder thresholds, NOT validated. They sit where the baseline scores of the three synthetic
# decks separate (Low up to about 0.11, Medium 0.06-0.24, High above 0.24), so the labels spread
# instead of everything being Low. Matching the synthetic labels is circular (same properties).
# Phase 3 replaces this rule with label-free percentile scoring against a reference set.
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


ARTIFACT_INT8 = "model_int8.onnx"
ARTIFACT_FP32 = "model_fp32.onnx"
ARTIFACT_LABELS = "labels.json"
ARTIFACT_FEATURE_SCHEMA = "feature_schema.json"
ARTIFACT_INFERENCE_CONFIG = "inference_config.json"
ARTIFACT_TOKENIZER = "tokenizer.json"
LABEL_ORDER: tuple[str, ...] = tuple(label.value for label in DifficultyLabel)  # ordinal order
MAX_LENGTH_LIMIT = 512

ARTIFACT_SPEC = f"""\
A trained-model folder (EDUSYNC_C2_MODEL_DIR) must contain:
  {ARTIFACT_INT8} or {ARTIFACT_FP32}: the ONNX graph (exactly one required; int8 is preferred
      when both exist)
  {ARTIFACT_LABELS}: JSON list {list(LABEL_ORDER)} in exactly this ordinal order (not alphabetical)
  {ARTIFACT_FEATURE_SCHEMA}: JSON object with schema_version (str, equal to
      registry.FEATURE_SCHEMA_VERSION), names (equal to registry.FEATURE_NAMES, in order) and
      hash (equal to registry.feature_schema_hash())
  {ARTIFACT_INFERENCE_CONFIG}: JSON object with max_length (int, 1..{MAX_LENGTH_LIMIT}),
      base_model (str), base_revision (str); optional trained_on (str)
  {ARTIFACT_TOKENIZER}: the non-empty tokenizer file
"""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"{path.name} is not readable valid JSON: {exc}") from exc


def _require_file(model_dir: Path, name: str) -> Path:
    path = model_dir / name
    if not path.is_file():
        raise ModelArtifactError(f"missing {name} in {model_dir}")
    return path


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


class OnnxFusionModel:
    """Fusion model (MiniLM text plus structured features) described by ONNX artifacts.

    It only validates the artifact folder against ``ARTIFACT_SPEC``; inference comes later.
    """

    def __init__(self, model_dir: Path, onnx_file: Path, version: str, trained_on: str | None):
        self.model_dir = model_dir
        self.onnx_file = onnx_file
        self.info = ModelInfo(
            name="onnx_fusion",
            version=version,
            kind=ModelKind.TODO,
            description="Artifacts validated; inference is not implemented until Phase 3",
            trained_on=trained_on,
        )

    @classmethod
    def from_artifacts(cls, model_dir: Path) -> "OnnxFusionModel":
        """Validate ``model_dir`` against ``ARTIFACT_SPEC``; raise ModelArtifactError if not."""
        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise ModelArtifactError(f"model directory does not exist: {model_dir}")

        int8, fp32 = model_dir / ARTIFACT_INT8, model_dir / ARTIFACT_FP32
        if int8.is_file():
            onnx_file = int8
        elif fp32.is_file():
            onnx_file = fp32
        else:
            raise ModelArtifactError(
                f"missing ONNX model: need {ARTIFACT_INT8} or {ARTIFACT_FP32} in {model_dir}"
            )

        labels = _read_json(_require_file(model_dir, ARTIFACT_LABELS))
        if labels != list(LABEL_ORDER):
            raise ModelArtifactError(
                f"{ARTIFACT_LABELS} must be exactly {list(LABEL_ORDER)} in ordinal order "
                f"(not alphabetical), got {labels!r}"
            )

        schema = _read_json(_require_file(model_dir, ARTIFACT_FEATURE_SCHEMA))
        version = schema.get("schema_version") if isinstance(schema, dict) else None
        if not _is_nonempty_str(version):
            raise ModelArtifactError(
                f"{ARTIFACT_FEATURE_SCHEMA} must be an object with a string schema_version"
            )
        if version != FEATURE_SCHEMA_VERSION:
            raise ModelArtifactError(
                f"{ARTIFACT_FEATURE_SCHEMA} schema_version {version!r} does not match the "
                f"current FEATURE_SCHEMA_VERSION {FEATURE_SCHEMA_VERSION!r}: the feature "
                "registry changed since training; retrain or restore the registry"
            )
        if schema.get("names") != list(FEATURE_NAMES):
            raise ModelArtifactError(
                f"{ARTIFACT_FEATURE_SCHEMA} names do not match registry.FEATURE_NAMES "
                "(different names or order): the feature registry changed since training"
            )
        if schema.get("hash") != feature_schema_hash():
            raise ModelArtifactError(
                f"{ARTIFACT_FEATURE_SCHEMA} hash {schema.get('hash')!r} does not match the "
                f"current feature_schema_hash() {feature_schema_hash()!r}: the feature registry "
                "changed since training; retrain or restore the registry"
            )

        config = _read_json(_require_file(model_dir, ARTIFACT_INFERENCE_CONFIG))
        if not isinstance(config, dict):
            raise ModelArtifactError(f"{ARTIFACT_INFERENCE_CONFIG} must be a JSON object")
        max_length = config.get("max_length")
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or not 1 <= max_length <= MAX_LENGTH_LIMIT
        ):
            raise ModelArtifactError(
                f"{ARTIFACT_INFERENCE_CONFIG} max_length must be an int in 1..{MAX_LENGTH_LIMIT}, "
                f"got {max_length!r}"
            )
        for key in ("base_model", "base_revision"):
            if not _is_nonempty_str(config.get(key)):
                raise ModelArtifactError(
                    f"{ARTIFACT_INFERENCE_CONFIG} {key} must be a non-empty string"
                )
        trained_on = config.get("trained_on")
        if trained_on is not None and not isinstance(trained_on, str):
            raise ModelArtifactError(f"{ARTIFACT_INFERENCE_CONFIG} trained_on must be a string")

        tokenizer = _require_file(model_dir, ARTIFACT_TOKENIZER)
        if tokenizer.stat().st_size == 0:
            raise ModelArtifactError(f"{ARTIFACT_TOKENIZER} is empty")

        return cls(model_dir, onnx_file, version, trained_on)

    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]:
        raise NotImplementedError(
            "Phase 3: ONNX inference for the fusion model is not implemented yet"
        )


MODEL_NAMES: tuple[str, ...] = ("baseline", "minilm")


def create_model(name: str = "baseline", model_dir: Path | None = None) -> DifficultyModel:
    """Model factory: ``baseline`` or ``minilm`` (artifact folder required)."""
    if name == "baseline":
        return RuleBaselineModel()
    if name == "minilm":
        if model_dir is None:
            raise ModelNotAvailableError(
                "EDUSYNC_C2_MODEL=minilm needs EDUSYNC_C2_MODEL_DIR (Phase 3 artifacts)"
            )
        return OnnxFusionModel.from_artifacts(model_dir)
    raise ModelNotAvailableError(f"unknown model {name!r}; valid names: {', '.join(MODEL_NAMES)}")
