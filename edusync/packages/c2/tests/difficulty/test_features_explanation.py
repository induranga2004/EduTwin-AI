"""Group C: explanation and example support."""

import pytest

from edusync_c2.difficulty.features.explanation import explanation_features
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
)

NAMES = list(features_of(FeatureGroup.EXPLANATION))
SEMANTIC = (
    "expl.semantic_definition_count",
    "expl.semantic_example_count",
    "expl.semantic_support_score",
)
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
    for name in SEMANTIC:  # no scorer: unavailable, never a regex value or a mocked 0.0
        assert d[name].status is FeatureStatus.UNAVAILABLE
        assert d[name].value is None
        assert d[name].source == "semantic:off"
        assert d[name].note == "semantic scorer is off"


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
    assert all(f.note == "concept not found in materials" for f in fs)


def test_no_slides_with_a_scorer_does_not_score(make_scorer, keyword_encoder):
    scorer = make_scorer()
    calls = keyword_encoder.calls
    fs = explanation_features(C, [], True, scorer, "minilm:abcd1234")
    assert all(f.status is FeatureStatus.UNAVAILABLE for f in fs)
    assert keyword_encoder.calls == calls


def test_semantic_unavailable_scorer(make_slide):
    d = vals(
        explanation_features(
            C, [make_slide(1, "DEFN a sentence here.")], True, None, "minilm:unavailable"
        )
    )
    for name in SEMANTIC:
        assert d[name].status is FeatureStatus.UNAVAILABLE
        assert d[name].value is None
        assert d[name].source == "semantic:minilm"
        assert d[name].note == "semantic scorer unavailable"


def test_semantic_counts_and_support(make_slide, make_scorer):
    text = (
        "DEFN first definition sentence here.\nDEFN second definition sentence here.\n"
        "EXMP an example sentence here.\nplain filler words go here"
    )
    s = make_slide(1, text, title="T")
    s.notes = "DEFN a note definition sentence."
    d = vals(explanation_features(C, [s], True, make_scorer(), "minilm:abcd1234"))
    assert list(d) == NAMES
    assert d["expl.semantic_definition_count"].value == 3
    assert d["expl.semantic_example_count"].value == 1
    assert d["expl.semantic_support_score"].value == 1.0
    for name in SEMANTIC:
        assert d[name].status is FeatureStatus.COMPUTED
        assert d[name].source == "semantic:minilm"
        assert d[name].note == ""


@pytest.mark.parametrize(
    ("text", "support"),
    [
        ("plain filler words go here", 0.0),
        ("DEFN a definition sentence here.", 0.5),
        ("EXMP an example sentence here.", 0.5),
    ],
)
def test_semantic_support_levels(make_slide, make_scorer, text, support):
    d = vals(explanation_features(C, [make_slide(1, text)], True, make_scorer(), "minilm:x"))
    assert d["expl.semantic_support_score"].value == support


def test_semantic_title_sentence_is_scored_once(make_slide, make_scorer):
    title = "DEFN title that is a definition sentence"
    s = make_slide(1, f"{title}\nbody filler words here", title=title)
    d = vals(explanation_features(C, [s], True, make_scorer(), "minilm:x"))
    assert d["expl.semantic_definition_count"].value == 1


def test_semantic_budget_is_noted(make_slide, make_scorer):
    text = "DEFN one definition sentence here.\nDEFN two definition sentence here."
    scorer = make_scorer(max_sentences=1).for_request()
    d = vals(explanation_features(C, [make_slide(1, text)], True, scorer, "minilm:x"))
    assert d["expl.semantic_definition_count"].value == 1
    assert d["expl.semantic_definition_count"].note.startswith("sentence budget reached")


def test_regex_counts_are_independent_of_the_scorer(make_slide, make_scorer):
    s = make_slide(1, "X is defined as a thing. For example, here.")
    off = vals(explanation_features(C, [s], True))
    on = vals(explanation_features(C, [s], True, make_scorer(), "minilm:x"))
    for name in ("expl.definition_count", "expl.example_count", "expl.step_count"):
        assert off[name] == on[name]


def test_title_cues_are_counted_once(make_slide):
    # SlideRecord.text already contains the title, so a cue in the title must count once.
    title = "Step 1: for example Foo is defined as bar"
    s = make_slide(1, f"{title}\nbody", title=title)
    d = vals(explanation_features(C, [s], True))
    assert d["expl.step_count"].value == 1
    assert d["expl.example_count"].value == 1
    assert d["expl.definition_count"].value == 1
