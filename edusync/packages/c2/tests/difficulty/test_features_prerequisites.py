"""Group B: prerequisite features."""

from edusync_c2.difficulty.features.prerequisites import prerequisite_features
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
M = PrerequisiteMetrics(
    direct_count=2,
    chain_depth=2,
    transitive_count=3,
    mean_confidence=0.75,
    curriculum_defined_share=0.5,
)


def test_v0_graph_unavailable():
    fs = prerequisite_features(C, None, False, REAL)
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.UNAVAILABLE and f.value is None for f in fs)
    assert "waiting for C1" in fs[0].note


def test_unknown_concept():
    fs = prerequisite_features(C, None, True, REAL)
    assert all(f.status is FeatureStatus.UNAVAILABLE and "unknown" in f.note for f in fs)


def test_real_graph_computed():
    fs = prerequisite_features(C, M, True, REAL)
    assert [f.name for f in fs] == NAMES
    assert [f.value for f in fs] == [2.0, 2.0, 3.0, 0.75, 0.5]
    assert all(f.status is FeatureStatus.COMPUTED for f in fs)


def test_mock_graph_mocked():
    fs = prerequisite_features(C, M, True, MOCK)
    assert all(f.status is FeatureStatus.MOCKED and f.value is not None for f in fs)
    assert all("mock" in f.note for f in fs)


def test_root_concept():
    root = PrerequisiteMetrics(direct_count=0, chain_depth=0, transitive_count=0)
    fs = prerequisite_features(C, root, True, REAL)
    assert [f.status for f in fs[:3]] == [FeatureStatus.COMPUTED] * 3
    assert [f.value for f in fs[:3]] == [0.0, 0.0, 0.0]
    assert [f.status for f in fs[3:]] == [FeatureStatus.UNAVAILABLE] * 2
    assert fs[3].note == "no direct prerequisites"


def test_direct_but_no_confidence():
    m = PrerequisiteMetrics(direct_count=1, chain_depth=1, transitive_count=1)
    fs = prerequisite_features(C, m, True, REAL)
    assert fs[3].status is FeatureStatus.UNAVAILABLE
