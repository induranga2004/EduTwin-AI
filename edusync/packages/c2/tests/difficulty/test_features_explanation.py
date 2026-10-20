"""Group C: explanation and example support."""

from edusync_c2.difficulty.features.explanation import explanation_features
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
)

NAMES = list(features_of(FeatureGroup.EXPLANATION))
C = ConceptRecord(id="x", label="X", alignment_status=AlignmentStatus.ALIGNED)


def vals(fs):
    return {f.name: f for f in fs}


def test_counts(make_slide):
    text = (
        "X is defined as a thing. It refers to stuff and is called Y; it means Z.\n"
        "For example, e.g. this, such as that. Example: here.\n"
        "Step 1: start\nstep 2: go\n1. one\n2) two\nFirst, we do a, then b."
    )
    s = make_slide(1, text, title="T", pictures=2, charts=1)
    s.notes = "For instance, notes count. It is a model."
    d = vals(explanation_features(C, [s], True))
    assert list(d) == NAMES
    assert d["expl.definition_count"].value == 5  # defined as, refers to, called, means, is a
    assert d["expl.example_count"].value == 5  # for example, e.g., such as, example:, for instance
    assert d["expl.step_count"].value == 5
    assert d["expl.diagram_support"].value == 3
    assert d["expl.semantic_support_score"].status is FeatureStatus.MOCKED
    assert d["expl.semantic_support_score"].value == 0.0


def test_pdf_diagram_unavailable(make_slide):
    d = vals(explanation_features(C, [make_slide(1, "plain", pictures=None, charts=None)], False))
    assert d["expl.diagram_support"].status is FeatureStatus.UNAVAILABLE
    assert "PDF" in d["expl.diagram_support"].note
    assert d["expl.example_count"].value == 0


def test_none_counts_treated_as_zero(make_slide):
    d = vals(explanation_features(C, [make_slide(1, "x", pictures=None, charts=None)], True))
    assert d["expl.diagram_support"].value == 0


def test_no_slides():
    fs = explanation_features(C, [], True)
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.UNAVAILABLE for f in fs)
