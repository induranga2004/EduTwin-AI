"""Group B: prerequisite features."""

from edusync_c2.difficulty.features.prerequisites import SOURCE, prerequisite_features
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    InputSource,
    PrerequisiteMetrics,
)

NAMES = list(features_of(FeatureGroup.PREREQUISITES))
C = ConceptRecord(id="gd", label="GD", alignment_status=AlignmentStatus.ALIGNED)
REAL = InputSource(name="c1_graph", kind="real")
MOCK = InputSource(name="c1_graph", kind="mock")
BUILT = NAMES


M = PrerequisiteMetrics(
    direct_count=2,
    chain_depth=2,
    transitive_count=3,
    mean_confidence=0.75,
    curriculum_defined_share=0.5,
    dependents_count=4,
    weighted_depth=1.5,
)


def test_v0_graph_unavailable():
    fs = prerequisite_features(C, None, False, REAL)
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.UNAVAILABLE and f.value is None for f in fs)
    assert all("waiting for C1" in f.note for f in fs if f.name in BUILT)


def test_unknown_concept():
    fs = prerequisite_features(C, None, True, REAL)
    built = [f for f in fs if f.name in BUILT]
    assert all(f.status is FeatureStatus.UNAVAILABLE and "unknown" in f.note for f in built)


def test_real_graph_computed():
    fs = prerequisite_features(C, M, True, REAL)
    assert [f.name for f in fs] == NAMES
    assert {f.name: f.value for f in fs} == {
        "prereq.direct_count": 2.0,
        "prereq.chain_depth": 2.0,
        "prereq.transitive_count": 3.0,
        "prereq.mean_confidence": 0.75,
        "prereq.curriculum_defined_share": 0.5,
        "prereq.weighted_depth": 1.5,
        "prereq.dependents_count": 4.0,
    }
    assert all(f.status is FeatureStatus.COMPUTED and f.source == SOURCE for f in fs)


def test_mock_graph_mocked():
    fs = prerequisite_features(C, M, True, MOCK)
    assert all(f.status is FeatureStatus.MOCKED and f.value is not None for f in fs)
    assert all("mock" in f.note and f.source == "mock:c1_graph" for f in fs)


def test_root_concept():
    root = PrerequisiteMetrics(direct_count=0, chain_depth=0, transitive_count=0)
    fs = prerequisite_features(C, root, True, REAL)
    by = {f.name: f for f in fs}
    computed = {
        "prereq.direct_count",
        "prereq.chain_depth",
        "prereq.transitive_count",
        "prereq.weighted_depth",
        "prereq.dependents_count",
    }
    assert {n for n, f in by.items() if f.status is FeatureStatus.COMPUTED} == computed
    assert all(by[n].value == 0.0 for n in computed)
    assert by["prereq.mean_confidence"].status is FeatureStatus.UNAVAILABLE
    assert by["prereq.curriculum_defined_share"].status is FeatureStatus.UNAVAILABLE
    assert by["prereq.mean_confidence"].note == "no direct prerequisites"


def test_direct_but_no_confidence():
    m = PrerequisiteMetrics(direct_count=1, chain_depth=1, transitive_count=1)
    fs = prerequisite_features(C, m, True, REAL)
    by = {f.name: f for f in fs}
    assert by["prereq.mean_confidence"].status is FeatureStatus.UNAVAILABLE
    assert by["prereq.weighted_depth"].status is FeatureStatus.COMPUTED
