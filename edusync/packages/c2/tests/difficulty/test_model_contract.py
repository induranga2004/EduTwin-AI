"""The model contract suite applied to the baseline, a conforming double and violating doubles.

``assert_model_contract`` lives here (not in an importable helper module) because the test
folders are not packages under ``--import-mode=importlib``. Phase 3 can import it by copying it
or by moving it to a package once a shared test-utils module exists.
"""

import math

import pytest

from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.fusion import build_fusion_input
from edusync_c2.difficulty.models import DifficultyModel, RuleBaselineModel
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    DifficultyLabel,
    FeatureStatus,
    FusionFeature,
    FusionInput,
    InputSource,
    ModelOutput,
    PrerequisiteMetrics,
)

REAL = InputSource(name="c1_graph", kind="real")
METRICS = PrerequisiteMetrics(
    direct_count=2,
    chain_depth=1,
    transitive_count=2,
    mean_confidence=0.8,
    curriculum_defined_share=0.5,
)


@pytest.fixture
def inputs(deck, graph_v1):
    slides_by_index = {s.index: s for s in deck.slides}
    concepts = [
        ConceptRecord(
            id=n.id,
            label=n.label,
            description=n.description,
            frequency=n.frequency,
            slide_indexes=[i],
            alignment_status=AlignmentStatus.ALIGNED,
        )
        for i, n in enumerate(graph_v1.graph.nodes, start=1)
    ]
    out = []
    for c in concepts:
        features = compute_all_features(
            c,
            all_concepts=concepts,
            slides_by_index=slides_by_index,
            structure_available=True,
            metrics=METRICS,
            graph_has_prerequisite_edges=True,
            graph_source=REAL,
        )
        out.append(build_fusion_input(c, features, [slides_by_index[c.slide_indexes[0]]]))
    # a concept with every feature missing must not make a model raise
    empty = [
        FusionFeature(name=f.name, value=None, status=FeatureStatus.UNAVAILABLE, missing=True)
        for f in out[0].features
    ]
    out.append(FusionInput(concept_id="empty", text="", features=empty))
    return out


PROBABILITY_TOLERANCE = 1e-6


def assert_output_valid(output: ModelOutput) -> None:
    assert isinstance(output.label, DifficultyLabel)
    if output.score is not None:
        assert 0.0 <= output.score <= 1.0, f"score {output.score} outside [0, 1]"
    if output.probabilities is not None:
        assert set(output.probabilities) == set(DifficultyLabel), "probabilities need 3 labels"
        assert all(p >= 0.0 for p in output.probabilities.values()), "negative probability"
        total = sum(output.probabilities.values())
        assert math.isclose(total, 1.0, abs_tol=PROBABILITY_TOLERANCE), f"sum is {total}"
    # ModelOutput has no confidence field; a model must not smuggle one in.
    assert not hasattr(output, "confidence")


def assert_model_contract(model: DifficultyModel, inputs: list[FusionInput]) -> None:
    """Raise AssertionError when ``model`` breaks the contract on ``inputs``."""
    assert model.info.name
    assert model.predict([]) == []
    first = model.predict(inputs)
    assert len(first) == len(inputs), "one output (or None) per input"
    for output in first:
        if output is not None:
            assert_output_valid(output)
    assert model.predict(inputs) == first, "predictions must be deterministic"


class _Base:
    info = RuleBaselineModel.info

    def predict(self, inputs):
        return [self._one(i) for i in inputs]


class GoodProbabilityModel(_Base):
    def _one(self, item):
        probs = {
            DifficultyLabel.LOW: 0.2,
            DifficultyLabel.MEDIUM: 0.5,
            DifficultyLabel.HIGH: 0.3,
        }
        return ModelOutput(label=DifficultyLabel.MEDIUM, score=0.5, probabilities=probs)


class NoneModel(_Base):
    def _one(self, item):
        return None


def test_baseline_satisfies_contract(inputs):
    assert_model_contract(RuleBaselineModel(), inputs)
    outputs = RuleBaselineModel().predict(inputs)
    assert outputs[-1] is None  # all features missing -> no prediction, no error
    assert any(o is not None for o in outputs)


def test_probability_double_satisfies_contract(inputs):
    assert_model_contract(GoodProbabilityModel(), inputs)


def test_model_returning_none_satisfies_contract(inputs):
    assert_model_contract(NoneModel(), inputs)


def test_bad_probability_sum_fails(inputs):
    class BadSum(_Base):
        def _one(self, item):
            probs = dict.fromkeys(DifficultyLabel, 0.4)
            return ModelOutput(label=DifficultyLabel.LOW, probabilities=probs)

    with pytest.raises(AssertionError, match="sum"):
        assert_model_contract(BadSum(), inputs)


def test_missing_label_in_probabilities_fails(inputs):
    class TwoLabels(_Base):
        def _one(self, item):
            probs = {DifficultyLabel.LOW: 0.5, DifficultyLabel.HIGH: 0.5}
            return ModelOutput(label=DifficultyLabel.LOW, probabilities=probs)

    with pytest.raises(AssertionError, match="3 labels"):
        assert_model_contract(TwoLabels(), inputs)


def test_wrong_length_fails(inputs):
    class Short(_Base):
        def predict(self, items):
            return []

    with pytest.raises(AssertionError, match="one output"):
        assert_model_contract(Short(), inputs)


def test_non_empty_result_for_empty_input_fails(inputs):
    class Eager(_Base):
        def predict(self, items):
            return [None]

    with pytest.raises(AssertionError):
        assert_model_contract(Eager(), inputs)


def test_non_deterministic_model_fails(inputs):
    class Flaky(_Base):
        calls = 0

        def predict(self, items):
            Flaky.calls += 1
            label = DifficultyLabel.LOW if Flaky.calls % 2 else DifficultyLabel.HIGH
            return [ModelOutput(label=label) for _ in items]

    with pytest.raises(AssertionError, match="deterministic"):
        assert_model_contract(Flaky(), inputs)


def test_invented_confidence_attribute_fails(inputs):
    class WithConfidence(ModelOutput):
        confidence: float = 0.9

    class Confident(_Base):
        def _one(self, item):
            return WithConfidence(label=DifficultyLabel.LOW)

    with pytest.raises(AssertionError):
        assert_model_contract(Confident(), inputs)


def test_negative_probability_fails(inputs):
    class Negative(_Base):
        def _one(self, item):
            probs = {
                DifficultyLabel.LOW: -0.5,
                DifficultyLabel.MEDIUM: 1.0,
                DifficultyLabel.HIGH: 0.5,
            }
            return ModelOutput(label=DifficultyLabel.LOW, probabilities=probs)

    with pytest.raises(AssertionError, match="negative"):
        assert_model_contract(Negative(), inputs)
