"""Fusion input, vector and summary."""

from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.fusion import build_fusion_input, summarize_features, to_vector
from edusync_c2.difficulty.registry import FEATURE_NAMES, FEATURE_SPECS, features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    InputSource,
    PrerequisiteMetrics,
)

PLANNED = {s.name for s in FEATURE_SPECS if s.maturity == "planned"}
MOCK = InputSource(name="c1_graph", kind="mock")
M = PrerequisiteMetrics(
    direct_count=1,
    chain_depth=1,
    transitive_count=1,
    mean_confidence=0.5,
    curriculum_defined_share=1.0,
)


def setup(deck):
    c = ConceptRecord(
        id="gd",
        label="Gradient Descent",
        description="About GD.",
        frequency=6,
        slide_indexes=[1, 2],
        alignment_status=AlignmentStatus.ALIGNED,
    )
    sl = [s for s in deck.slides if s.index in (1, 2)]
    fs = compute_all_features(
        c,
        all_concepts=[c],
        slides_by_index={s.index: s for s in deck.slides},
        structure_available=True,
        metrics=M,
        graph_has_prerequisite_edges=True,
        graph_source=MOCK,
    )
    return c, sl, fs


def test_order_mask_and_values(deck):
    c, sl, fs = setup(deck)
    fi = build_fusion_input(c, list(reversed(fs)), sl)
    assert [f.name for f in fi.features] == list(FEATURE_NAMES)
    vals, mask = to_vector(fi)
    assert len(vals) == len(mask) == len(FEATURE_NAMES)
    for f, v, m in zip(fi.features, vals, mask, strict=True):
        assert m == int(f.missing)
        if f.missing:
            assert v == 0.0
            assert f.value is None
    miss = {f.name for f in fi.features if f.missing}
    assert "dens.visual_complexity" in miss
    assert "prereq.direct_count" in miss
    assert "lang.avg_word_length" not in miss


def test_use_mocked(deck):
    c, sl, fs = setup(deck)
    fi = build_fusion_input(c, fs, sl, use_mocked=True)
    miss = {f.name for f in fi.features if f.missing}
    assert miss == PLANNED


def test_missing_feature_filled(deck):
    c, sl, fs = setup(deck)
    fi = build_fusion_input(c, fs[:3], sl)
    assert len(fi.features) == len(FEATURE_NAMES)
    assert fi.features[10].status is FeatureStatus.UNAVAILABLE
    assert fi.features[10].missing


def test_text_truncation_keeps_header(deck):
    c, sl, _ = setup(deck)
    fi = build_fusion_input(c, [], sl, max_text_chars=10)
    assert fi.text == "Gradient Descent. About GD.\n"
    full = build_fusion_input(c, [], sl)
    assert full.text.startswith("Gradient Descent. About GD.\nGradient Descent")
    cut = build_fusion_input(c, [], sl, max_text_chars=60)
    assert len(cut.text) == 60
    c.description = None
    assert build_fusion_input(c, [], []).text == "Gradient Descent. \n"


def test_summary(deck):
    _, _, fs = setup(deck)
    s = summarize_features(fs)
    assert set(s) == set(FeatureGroup)
    assert sum(g.computed + g.mocked + g.unavailable for g in s.values()) == len(FEATURE_NAMES)
    assert s[FeatureGroup.PREREQUISITES].mocked == len(features_of(FeatureGroup.PREREQUISITES))
    assert s[FeatureGroup.PREREQUISITES].unavailable == 0  # group B is fully built
    assert s[FeatureGroup.LANGUAGE].mocked == 0
    assert s[FeatureGroup.LANGUAGE].computed == 8
    assert s[FeatureGroup.DENSITY].unavailable == 6  # the six planned density features
    empty = summarize_features([])
    assert len(empty) == 4
    assert empty[FeatureGroup.LANGUAGE].computed == 0
