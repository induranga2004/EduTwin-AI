import pytest

from edusync_c2.difficulty.models import (
    LOW_MAX,
    MEDIUM_MAX,
    DifficultyModel,
    MiniLMFusionModel,
    RuleBaselineModel,
)
from edusync_c2.difficulty.registry import FEATURE_SPECS
from edusync_c2.difficulty.schemas import (
    DifficultyLabel,
    FeatureStatus,
    FusionFeature,
    FusionInput,
    ModelKind,
    ModelOutput,
)


def make_input(values: dict[str, float], *, mocked: tuple[str, ...] = ()) -> FusionInput:
    feats = []
    for spec in FEATURE_SPECS:
        if spec.name in mocked:
            feats.append(
                FusionFeature(name=spec.name, value=None, status=FeatureStatus.MOCKED, missing=True)
            )
        elif spec.name in values:
            feats.append(
                FusionFeature(
                    name=spec.name,
                    value=values[spec.name],
                    status=FeatureStatus.COMPUTED,
                    missing=False,
                )
            )
        else:
            feats.append(
                FusionFeature(
                    name=spec.name, value=None, status=FeatureStatus.UNAVAILABLE, missing=True
                )
            )
    return FusionInput(concept_id="c", text="t", features=feats)


def predict(values, **kw):
    return RuleBaselineModel().predict([make_input(values, **kw)])[0]


THREE = {"lang.lexical_score": 0.5, "dens.words_per_slide": 60.0, "dens.formula_presence": 0.5}
HIGH3 = {"lang.lexical_score": 1.0, "dens.code_presence": 1.0, "dens.formula_presence": 1.0}


def test_info_and_no_probabilities():
    m = RuleBaselineModel()
    assert m.info.kind is ModelKind.BASELINE
    assert m.info.trained_on is None
    out = predict(THREE)
    assert out.probabilities is None
    # weights .25 (0.5) + .15 (0.5) + .10 (0.5) over .50
    assert out.score == pytest.approx(0.5)


def test_fewer_than_three_components_gives_none():
    assert predict({"lang.lexical_score": 0.5, "dens.words_per_slide": 60.0}) is None
    assert predict({}) is None
    support_only = {"lang.lexical_score": 0.5, "expl.example_count": 1, "expl.step_count": 1}
    assert predict(support_only) is None


def test_mocked_and_unavailable_ignored():
    values = {**THREE, "prereq.direct_count": 4.0}
    out = predict(values, mocked=("prereq.direct_count",))
    assert out.score == pytest.approx(0.5)
    assert not any("prereq.direct_count" in line for line in out.explanation)
    assert predict(THREE, mocked=("dens.formula_presence",)) is None


@pytest.mark.parametrize(
    ("name", "raw", "norm"),
    [
        ("dens.words_per_slide", 60.0, 0.5),
        ("dens.words_per_slide", 500.0, 1.0),
        ("dens.concepts_per_slide", 0.5, 0.0),
        ("dens.concepts_per_slide", 3.0, 0.5),
        ("dens.concepts_per_slide", 9.0, 1.0),
        ("prereq.direct_count", 2.0, 0.5),
        ("prereq.chain_depth", 9.0, 1.0),
        ("dens.code_presence", 1.0, 1.0),
    ],
)
def test_normalisation(name, raw, norm):
    values = {"lang.lexical_score": 0.0, "dens.formula_presence": 0.0, "dens.words_per_slide": 0.0}
    values[name] = raw
    if len(values) < 3:
        values["prereq.chain_depth"] = 0.0
    out = predict(values)
    line = next(ln for ln in out.explanation if ln.startswith(name))
    assert f"= {norm:.2f} (computed" in line


def test_weights_in_weighted_mean():
    values = {"lang.lexical_score": 1.0, "dens.code_presence": 0.0, "dens.formula_presence": 0.0}
    assert predict(values).score == pytest.approx(0.25 / 0.40, abs=1e-4)


def test_support_credit():
    full = {"expl.definition_count": 3, "expl.example_count": 2, "expl.step_count": 1}
    assert predict(HIGH3).score == 1.0
    credited = predict({**HIGH3, **full})
    assert credited.score == pytest.approx(0.75)
    partial = predict({**THREE, "expl.example_count": 1})
    assert partial.score == pytest.approx(0.5 - 0.25 / 3, abs=1e-4)
    assert any("support credit" in ln for ln in credited.explanation)


def test_score_clamped_at_zero():
    out = predict(
        {
            "lang.lexical_score": 0.0,
            "dens.words_per_slide": 0.0,
            "dens.formula_presence": 0.0,
            "expl.definition_count": 1,
            "expl.example_count": 1,
            "expl.step_count": 1,
        }
    )
    assert out.score == 0.0
    assert out.label is DifficultyLabel.LOW


@pytest.mark.parametrize(
    ("lex", "label"),
    [
        (0.0, DifficultyLabel.LOW),
        (LOW_MAX - 0.01, DifficultyLabel.LOW),
        (LOW_MAX, DifficultyLabel.MEDIUM),
        (MEDIUM_MAX - 0.01, DifficultyLabel.MEDIUM),
        (MEDIUM_MAX, DifficultyLabel.HIGH),
        (1.0, DifficultyLabel.HIGH),
    ],
)
def test_label_thresholds(lex, label):
    values = {"lang.lexical_score": lex, "dens.code_presence": lex, "dens.formula_presence": lex}
    assert predict(values).label is label


def test_explanation_content():
    out = predict(THREE)
    assert out.explanation[0].startswith("lang.lexical_score = 0.50 (computed, weight 0.25)")
    assert len(out.explanation) == 3 + 2
    assert "baseline, not trained" in out.explanation[-1]
    assert f"Low < {LOW_MAX}" in out.explanation[-1]
    assert predict(THREE) == out


def test_todo_model_raises():
    m = MiniLMFusionModel()
    assert m.info.kind is ModelKind.TODO
    with pytest.raises(NotImplementedError, match="Phase 3"):
        m.predict([])


def test_protocol_conformance_with_fake_model():
    class Fake:
        info = RuleBaselineModel.info

        def predict(self, inputs):
            return [ModelOutput(label=DifficultyLabel.LOW) for _ in inputs]

    model: DifficultyModel = Fake()
    assert model.predict([make_input({})])[0].label is DifficultyLabel.LOW
