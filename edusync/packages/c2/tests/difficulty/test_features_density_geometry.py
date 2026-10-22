"""Group D: slide geometry, fonts, diagram elements, table cells and visual complexity."""

from fractions import Fraction

import pytest

from edusync_c2.difficulty.features.density import (
    _VC_SATURATION,
    _VC_WEIGHTS,
    density_features,
)
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureStatus,
    InputSource,
)

STRUCTURE = (
    "dens.text_area_ratio",
    "dens.picture_area_ratio",
    "dens.min_font_pt",
    "dens.diagram_element_count",
    "dens.table_cell_count",
    "dens.visual_complexity",
)
REAL = InputSource(name="c1_graph", kind="real")
W, H = 9_144_000, 6_858_000  # 10 in x 7.5 in
AREA = W * H
SIZE = (W, H)


def run(slides, size=SIZE, structure=True):
    me = ConceptRecord(
        id="me",
        label="me",
        slide_indexes=[s.index for s in slides],
        alignment_status=AlignmentStatus.ALIGNED,
    )
    return {f.name: f for f in density_features(me, slides, [me], structure, REAL, size)}


def full_slide(make_slide, index=1, **kw):
    base = {
        "pictures": 2,
        "charts": 1,
        "shapes": 10,
        "smartart_count": 1,
        "diagram_element_count": 15,
        "table_cell_count": 20,
        "visual_area_emu": AREA // 2,
    }
    base.update(kw)
    return make_slide(index, "a", **base)


def test_ratio_means_are_taken_over_slides_and_clamped(make_slide):
    s1 = make_slide(1, "a", text_area_emu=AREA // 2, picture_area_emu=AREA // 4)
    s2 = make_slide(2, "b", text_area_emu=AREA * 3, picture_area_emu=-5)  # over and under range
    r = run([s1, s2])
    assert r["dens.text_area_ratio"].value == pytest.approx((0.5 + 1.0) / 2)
    assert r["dens.picture_area_ratio"].value == pytest.approx(0.125)
    assert r["dens.text_area_ratio"].status is FeatureStatus.COMPUTED
    assert r["dens.text_area_ratio"].source == "materials:pptx_structure"
    assert r["dens.text_area_ratio"].note == ""


def test_ratios_skip_slides_without_values(make_slide):
    s1 = make_slide(1, "a", text_area_emu=AREA // 4)
    s2 = make_slide(2, "b")  # unknown geometry: skipped, not counted as 0
    r = run([s1, s2])
    assert r["dens.text_area_ratio"].value == 0.25
    assert r["dens.picture_area_ratio"].status is FeatureStatus.UNAVAILABLE
    assert r["dens.picture_area_ratio"].value is None
    assert r["dens.picture_area_ratio"].note == "geometry unknown"
    assert r["dens.picture_area_ratio"].source == "materials:pptx_structure"


@pytest.mark.parametrize("size", [None, (0, H), (W, 0), (-1, -1)])
def test_ratios_need_a_positive_slide_size(make_slide, size):
    s = make_slide(1, "a", text_area_emu=1, picture_area_emu=1, visual_area_emu=1)
    r = run([s], size=size)
    for name in ("dens.text_area_ratio", "dens.picture_area_ratio"):
        assert r[name].status is FeatureStatus.UNAVAILABLE
        assert r[name].note == "slide size unknown"


def test_min_font_is_the_minimum_not_the_mean(make_slide):
    slides = [make_slide(i, "a", min_font_pt=v) for i, v in ((1, 18.0), (2, 10.5), (3, None))]
    f = run(slides)["dens.min_font_pt"]
    assert f.value == 10.5
    assert f.status is FeatureStatus.COMPUTED
    assert f.note == ""


def test_min_font_unavailable_without_an_explicit_size(make_slide):
    f = run([make_slide(1, "a")])["dens.min_font_pt"]
    assert f.status is FeatureStatus.UNAVAILABLE
    assert f.value is None
    assert f.note == "no explicit font size"


def test_counts_are_means_per_slide(make_slide):
    slides = [
        make_slide(1, "a", diagram_element_count=3, table_cell_count=12),
        make_slide(2, "b", diagram_element_count=0, table_cell_count=0),
        make_slide(3, "c"),  # unknown: skipped
    ]
    r = run(slides)
    assert r["dens.diagram_element_count"].value == 1.5
    assert r["dens.table_cell_count"].value == 6.0


def test_counts_unavailable_when_every_slide_is_unknown(make_slide):
    r = run([make_slide(1, "a")])
    for name in ("dens.diagram_element_count", "dens.table_cell_count"):
        assert r[name].status is FeatureStatus.UNAVAILABLE
        assert r[name].note == "geometry unknown"


def test_counts_and_font_do_not_need_the_slide_size(make_slide):
    s = make_slide(1, "a", diagram_element_count=4, table_cell_count=2, min_font_pt=9.0)
    r = run([s], size=None)
    assert r["dens.diagram_element_count"].value == 4.0
    assert r["dens.table_cell_count"].value == 2.0
    assert r["dens.min_font_pt"].value == 9.0


def test_lower_bound_note_when_shapes_were_unreadable(make_slide):
    s1 = make_slide(
        1,
        "a",
        text_area_emu=AREA // 2,
        picture_area_emu=0,
        diagram_element_count=2,
        table_cell_count=1,
        min_font_pt=12.0,
        geometry_unknown_count=2,
    )
    s2 = make_slide(2, "b", text_area_emu=AREA // 2, geometry_unknown_count=1)
    r = run([s1, s2])
    assert r["dens.text_area_ratio"].note == "lower bound: 3 shape(s) with unreadable geometry"
    assert r["dens.text_area_ratio"].status is FeatureStatus.COMPUTED
    assert r["dens.text_area_ratio"].value == 0.5
    # only the slides that contributed to a feature are counted
    assert r["dens.picture_area_ratio"].note == "lower bound: 2 shape(s) with unreadable geometry"
    assert r["dens.diagram_element_count"].note.startswith("lower bound: 2 ")
    assert r["dens.table_cell_count"].note.startswith("lower bound: 2 ")
    assert r["dens.min_font_pt"].note == "upper bound: 2 shape(s) with unreadable geometry"
    assert r["dens.min_font_pt"].value == 12.0


def test_no_note_when_nothing_was_unreadable(make_slide):
    s = make_slide(1, "a", text_area_emu=1, geometry_unknown_count=0)
    assert run([s])["dens.text_area_ratio"].note == ""


def test_visual_complexity_formula(make_slide):
    f = run([full_slide(make_slide)])["dens.visual_complexity"]
    # media 2+1+1 = 4 -> 1; diagram 15 -> 1; visual area 0.5; info 20+10 = 30 -> 1
    assert f.value == pytest.approx(0.35 + 0.25 + 0.25 * 0.5 + 0.15)
    assert f.status is FeatureStatus.COMPUTED
    assert f.source == "materials:pptx_structure"
    assert f.note == ""


def test_visual_complexity_saturates_and_means_over_slides(make_slide):
    a = full_slide(make_slide, 1, pictures=40, diagram_element_count=300, visual_area_emu=AREA * 2)
    b = full_slide(
        make_slide,
        2,
        pictures=0,
        charts=0,
        smartart_count=0,
        diagram_element_count=0,
        table_cell_count=0,
        shapes=0,
        visual_area_emu=0,
    )
    f = run([a, b])["dens.visual_complexity"]
    # media mean 21.5 and diagram mean 150 saturate; visual area (1 + 0) / 2; info (30 + 0) / 2
    assert f.value == pytest.approx(0.35 + 0.25 + 0.25 * 0.5 + 0.15 * 0.5)


def test_visual_complexity_is_zero_for_an_empty_slide(make_slide):
    s = make_slide(
        1,
        "a",
        pictures=0,
        shapes=0,
        smartart_count=0,
        diagram_element_count=0,
        table_cell_count=0,
        visual_area_emu=0,
    )
    f = run([s])["dens.visual_complexity"]
    assert f.value == 0.0
    assert f.status is FeatureStatus.COMPUTED


def test_visual_complexity_renormalises_without_slide_size(make_slide):
    f = run([full_slide(make_slide)], size=None)["dens.visual_complexity"]
    assert f.status is FeatureStatus.COMPUTED
    assert f.value == pytest.approx(1.0)  # every known component is saturated
    assert f.note == "unknown components: visual_area"


def test_visual_complexity_renormalises_over_the_known_components(make_slide):
    s = make_slide(1, "a", pictures=None, shapes=3, diagram_element_count=3)
    f = run([s])["dens.visual_complexity"]
    assert f.value == pytest.approx(3 / 15)  # only the diagram component is known
    assert f.note == "unknown components: media, visual_area, info"


def test_visual_complexity_partial_weights(make_slide):
    # media (2) and diagram (3) known; visual area and info unknown
    s = make_slide(
        1, "a", pictures=2, charts=0, smartart_count=0, shapes=3, diagram_element_count=3
    )
    f = run([s])["dens.visual_complexity"]
    expected = (0.35 * (2 / 4) + 0.25 * (3 / 15)) / (0.35 + 0.25)
    assert f.value == pytest.approx(expected)


def test_visual_complexity_skips_slides_per_component(make_slide):
    a = full_slide(make_slide, 1, visual_area_emu=AREA)
    b = make_slide(2, "b", pictures=None)  # nothing known on this slide
    f = run([a, b])["dens.visual_complexity"]
    assert f.value == pytest.approx(1.0)


def test_visual_complexity_unavailable_when_no_component_is_known(make_slide):
    s = make_slide(1, "a", pictures=None, shapes=None)
    f = run([s])["dens.visual_complexity"]
    assert f.status is FeatureStatus.UNAVAILABLE
    assert f.value is None
    assert f.note == "geometry unknown"
    assert run([s], size=None)["dens.visual_complexity"].note == "slide size unknown"


def test_visual_complexity_lower_bound_note(make_slide):
    s = full_slide(make_slide, geometry_unknown_count=4)
    f = run([s], size=None)["dens.visual_complexity"]
    assert f.status is FeatureStatus.COMPUTED
    assert f.note == (
        "lower bound: 4 shape(s) with unreadable geometry; unknown components: visual_area"
    )


def test_visual_complexity_constants():
    assert set(_VC_SATURATION) == {"media", "diagram", "info"}
    assert set(_VC_WEIGHTS) == {"media", "diagram", "visual_area", "info"}
    assert sum(_VC_WEIGHTS.values()) == 1


def test_pdf_structure_features_are_unavailable(make_slide):
    s = make_slide(1, "a b", shapes=None, pictures=None)
    r = run([s], size=None, structure=False)
    for name in STRUCTURE:
        assert r[name].status is FeatureStatus.UNAVAILABLE
        assert r[name].value is None
        assert r[name].source == "materials:pptx_structure"
        assert r[name].note == "PDF: structure not available"


def test_geometry_features_are_deterministic(make_slide):
    slides = [full_slide(make_slide, i, text_area_emu=i * 7919, min_font_pt=9.5) for i in (1, 2, 3)]
    first = run(slides)
    second = run(list(slides))
    assert {k: v.model_dump() for k, v in first.items()} == {
        k: v.model_dump() for k, v in second.items()
    }
    exact = sum((Fraction(i * 7919, AREA) for i in (1, 2, 3)), Fraction(0)) / 3
    assert first["dens.text_area_ratio"].value == float(exact)
