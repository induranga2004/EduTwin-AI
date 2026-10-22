"""Registry conformance of compute_all_features."""

import pytest

from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.registry import FEATURE_NAMES
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureStatus,
    InputSource,
    PrerequisiteMetrics,
)

REAL = InputSource(name="c1_graph", kind="real")
MOCK = InputSource(name="c1_graph", kind="mock")
M = PrerequisiteMetrics(
    direct_count=2,
    chain_depth=1,
    transitive_count=2,
    mean_confidence=0.8,
    curriculum_defined_share=0.5,
)


def concepts() -> list[ConceptRecord]:
    return [
        ConceptRecord(
            id="gd",
            label="Gradient Descent",
            frequency=6,
            slide_indexes=[1, 2],
            alignment_status=AlignmentStatus.ALIGNED,
        ),
        ConceptRecord(
            id="ghost",
            label="Ghost",
            frequency=1,
            slide_indexes=[],
            alignment_status=AlignmentStatus.NOT_FOUND,
        ),
    ]


def run(deck, c, **kw):
    args = {
        "all_concepts": concepts(),
        "slides_by_index": {s.index: s for s in deck.slides},
        "structure_available": True,
        "metrics": M,
        "graph_has_prerequisite_edges": True,
        "graph_source": REAL,
    }
    args.update(kw)
    return compute_all_features(c, **args)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"structure_available": False},
        {"graph_source": MOCK},
        {"metrics": None, "graph_has_prerequisite_edges": False},
    ],
)
def test_names_and_order(deck, which, kw):
    fs = run(deck, concepts()[which], **kw)
    assert [f.name for f in fs] == list(FEATURE_NAMES)
    assert len(fs) == len(FEATURE_NAMES)


def test_statuses_real_deck(deck):
    fs = {f.name: f for f in run(deck, concepts()[0])}
    assert fs["lang.avg_word_length"].status is FeatureStatus.COMPUTED
    assert fs["prereq.direct_count"].status is FeatureStatus.COMPUTED
    assert fs["expl.definition_count"].value >= 1
    assert fs["expl.diagram_support"].value == 1
    assert fs["dens.words_per_slide"].status is FeatureStatus.COMPUTED


def test_ghost_concept_unavailable(deck):
    fs = {f.name: f for f in run(deck, concepts()[1])}
    assert fs["lang.avg_word_length"].status is FeatureStatus.UNAVAILABLE
    assert fs["dens.concept_frequency"].status is FeatureStatus.COMPUTED


def test_deterministic(deck):
    assert run(deck, concepts()[0]) == run(deck, concepts()[0])
