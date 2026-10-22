"""Registry snapshot and per-group completeness.

The snapshot (golden/registry_snapshot.json) pins the feature schema version, hash and the
ordered (name, group) list. A trained model depends on these, so they may only change together
with FEATURE_SCHEMA_VERSION.
"""

import json
from pathlib import Path

import pytest

from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    feature_schema_hash,
    features_of,
)
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    InputSource,
    PrerequisiteMetrics,
)

SNAPSHOT_PATH = Path(__file__).parent / "golden" / "registry_snapshot.json"
REGEN = "python edusync/packages/c2/scripts/feature_report.py --write-snapshot <path>"
SNAPSHOT = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
SNAP_VERSION = SNAPSHOT["version"]
SNAP_FEATURES = SNAPSHOT["features"]
CURRENT = [[s.name, s.group.value] for s in FEATURE_SPECS]


def test_version_matches_snapshot():
    assert FEATURE_SCHEMA_VERSION == SNAP_VERSION, (
        f"FEATURE_SCHEMA_VERSION is {FEATURE_SCHEMA_VERSION!r} but the snapshot is for "
        f"{SNAP_VERSION!r}: regenerate the snapshot ({REGEN}) and commit it."
    )


def test_names_groups_and_order_match_snapshot():
    if CURRENT != SNAP_FEATURES and FEATURE_SCHEMA_VERSION == SNAP_VERSION:
        pytest.fail(
            "Feature names, groups or order changed without a version change: bump "
            f"FEATURE_SCHEMA_VERSION and regenerate the snapshot ({REGEN})."
        )
    assert CURRENT == SNAP_FEATURES, f"Registry differs from the snapshot; run {REGEN}."


def test_hash_matches_snapshot():
    assert feature_schema_hash() == SNAPSHOT["hash"]


def test_group_sizes():
    sizes = {g: len(features_of(g)) for g in FeatureGroup}
    assert sizes == {
        FeatureGroup.LANGUAGE: 8,
        FeatureGroup.PREREQUISITES: 7,
        FeatureGroup.EXPLANATION: 7,
        FeatureGroup.DENSITY: 13,
    }
    assert sum(sizes.values()) == len(FEATURE_NAMES) == 35


# Completeness: every registered feature is produced by exactly one processor output.


GD = ConceptRecord(
    id="gd",
    label="Gradient Descent",
    frequency=6,
    slide_indexes=[1, 2],
    alignment_status=AlignmentStatus.ALIGNED,
)
GHOST = ConceptRecord(
    id="ghost", label="Ghost", alignment_status=AlignmentStatus.NOT_FOUND, slide_indexes=[]
)
METRICS = PrerequisiteMetrics(
    direct_count=2,
    chain_depth=1,
    transitive_count=2,
    mean_confidence=0.8,
    curriculum_defined_share=0.5,
)
SCENARIOS = {
    "pptx": {},
    "pdf": {"structure_available": False},
    "mock_graph": {"graph_source": InputSource(name="c1_graph", kind="mock")},
    "no_edges": {"metrics": None, "graph_has_prerequisite_edges": False},
    "unknown_concept": {"metrics": None},
    "ghost": {"concept": GHOST},
}


def _compute(deck, **kw):
    args = {
        "concept": GD,
        "structure_available": True,
        "metrics": METRICS,
        "graph_has_prerequisite_edges": True,
        "graph_source": InputSource(name="c1_graph", kind="real"),
    }
    args.update(kw)
    concept = args.pop("concept")
    return compute_all_features(
        concept,
        all_concepts=[GD],
        slides_by_index={s.index: s for s in deck.slides},
        **args,
    )


@pytest.fixture
def computed(deck):
    return _compute(deck)


def test_compute_all_features_covers_the_registry_once(computed):
    assert [f.name for f in computed] == list(FEATURE_NAMES)
    assert len({f.name for f in computed}) == len(FEATURE_NAMES)
    for f, spec in zip(computed, FEATURE_SPECS, strict=True):
        assert f.group is spec.group


@pytest.mark.parametrize("group", list(FeatureGroup))
def test_each_group_is_complete(computed, group):
    assert [f.name for f in computed if f.group is group] == list(features_of(group))


def test_every_registered_feature_is_implemented():
    # Phase 2 built all 35 features: nothing is planned or a mock any more.
    assert {s.maturity for s in FEATURE_SPECS} == {"implemented"}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_built_features_are_never_todo(deck, scenario):
    by_name = {f.name: f for f in _compute(deck, **SCENARIOS[scenario])}
    for s in FEATURE_SPECS:
        assert not by_name[s.name].source.startswith("todo:")
