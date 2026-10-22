"""Group D: density and structure."""

import pytest

from edusync_c2.difficulty.features.density import density_features
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    InputSource,
)

NAMES = list(features_of(FeatureGroup.DENSITY))
REAL = InputSource(name="c1_graph", kind="real")
MOCK = InputSource(name="c1_graph", kind="mock")


def rec(cid: str, idx: list[int], freq: int = 0) -> ConceptRecord:
    return ConceptRecord(
        id=cid,
        label=cid,
        frequency=freq,
        slide_indexes=idx,
        alignment_status=AlignmentStatus.ALIGNED,
    )


def d(fs):
    return {f.name: f for f in fs}


def test_values(make_slide):
    s1 = make_slide(1, "a b c\nd e", levels=[0, 2], shapes=2, formula=True)
    s2 = make_slide(2, "f g", levels=[1], shapes=4, code=True)
    me = rec("me", [1, 2], freq=7)
    allc = [me, rec("o", [1]), rec("p", [3])]
    fs = density_features(me, [s1, s2], allc, True, REAL)
    assert [f.name for f in fs] == NAMES
    r = d(fs)
    assert r["dens.words_per_slide"].value == pytest.approx(3.5)
    assert r["dens.concepts_per_slide"].value == pytest.approx(1.5)
    assert r["dens.formula_presence"].value == 0.5
    assert r["dens.code_presence"].value == 0.5
    assert r["dens.max_bullet_depth"].value == 2
    assert r["dens.shape_count_mean"].value == 3
    assert r["dens.concept_frequency"].value == 7
    assert r["dens.concept_frequency"].status is FeatureStatus.COMPUTED
    assert r["dens.visual_complexity"].status is FeatureStatus.UNAVAILABLE
    assert "TODO" in r["dens.visual_complexity"].note


def test_mock_graph_frequency(make_slide):
    me = rec("me", [1], 3)
    r = d(density_features(me, [make_slide(1, "a")], [me], True, MOCK))
    assert r["dens.concept_frequency"].status is FeatureStatus.MOCKED
    assert r["dens.concept_frequency"].value == 3


def test_pdf_like(make_slide):
    me = rec("me", [1])
    s = make_slide(1, "a b", shapes=None, pictures=None)
    r = d(density_features(me, [s], [me], False, REAL))
    for n in ("dens.max_bullet_depth", "dens.shape_count_mean"):
        assert r[n].status is FeatureStatus.UNAVAILABLE
        assert "PDF" in r[n].note
    assert r["dens.words_per_slide"].status is FeatureStatus.COMPUTED


def test_unknown_shape_count_and_no_levels(make_slide):
    me = rec("me", [1])
    s = make_slide(1, "a b", shapes=None)
    s.paragraphs = []
    r = d(density_features(me, [s], [me], True, REAL))
    assert r["dens.shape_count_mean"].status is FeatureStatus.UNAVAILABLE
    assert r["dens.max_bullet_depth"].value == 0


def test_no_slides_keeps_frequency():
    me = rec("me", [], 5)
    fs = density_features(me, [], [me], True, REAL)
    assert [f.name for f in fs] == NAMES
    st = {f.name: f.status for f in fs}
    assert st["dens.concept_frequency"] is FeatureStatus.COMPUTED
    assert (
        sum(s is FeatureStatus.UNAVAILABLE for s in st.values()) == 12
    )  # 6 built + visual_complexity + 5 other planned


@pytest.mark.parametrize("structure", [True, False])
def test_planned_structure_features_are_todo(make_slide, structure):
    me = rec("me", [1])
    fs = d(density_features(me, [make_slide(1, "a b")], [me], structure, REAL))
    for name in (
        "dens.text_area_ratio",
        "dens.picture_area_ratio",
        "dens.min_font_pt",
        "dens.diagram_element_count",
        "dens.table_cell_count",
    ):
        assert fs[name].status is FeatureStatus.UNAVAILABLE
        assert fs[name].value is None
        assert fs[name].source == f"todo:{name}"
