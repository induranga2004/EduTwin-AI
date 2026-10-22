"""Tests for the synthetic data builders (round trip through the real parser)."""

import csv
import io

import pytest

from edusync_c2.difficulty.materials import detect_code, detect_formula, parse_materials
from edusync_c2.difficulty.schemas import C1GraphResponse, DifficultyLabel
from edusync_c2.difficulty.synthetic import (
    ConceptSpec,
    build_c1_graph,
    build_labels_csv,
    build_pdf,
    build_pptx,
    make_deck_spec,
    synthetic_label,
)


def _concept(**overrides) -> ConceptSpec:
    base = {
        "label": "Alpha Beta",
        "definition": False,
        "examples": 0,
        "steps": 0,
        "words": 50,
        "formula": False,
        "code": False,
        "picture": False,
        "bullet_depth": 0,
        "prerequisites": [],
    }
    return ConceptSpec(**{**base, **overrides})


def test_spec_is_deterministic_and_acyclic() -> None:
    a, b = make_deck_spec(7, 10), make_deck_spec(7, 10)
    assert a == b
    assert a != make_deck_spec(8, 10)
    assert a.lecture_id == "synthetic-7"
    assert make_deck_spec(7, 3, lecture_id="x").lecture_id == "x"
    seen: list[str] = []
    for c in a.concepts:
        assert set(c.prerequisites) <= set(seen)
        seen.append(c.label)
    assert len({c.label.casefold() for c in a.concepts}) == 10


def test_spec_rejects_bad_size() -> None:
    with pytest.raises(ValueError, match="n_concepts"):
        make_deck_spec(1, 0)


def test_properties_vary_widely() -> None:
    concepts = [c for s in range(10) for c in make_deck_spec(s, 10).concepts]
    assert {c.formula for c in concepts} == {True, False}
    assert {c.code for c in concepts} == {True, False}
    assert {c.picture for c in concepts} == {True, False}
    assert {c.bullet_depth for c in concepts} == {0, 1, 2, 3}
    assert max(len(c.prerequisites) for c in concepts) == 3


def test_pptx_round_trip() -> None:
    spec = make_deck_spec(3, 12)
    doc = parse_materials(build_pptx(spec), "x.pptx")
    assert doc.slide_count == len(spec.concepts) + 2
    assert doc.slides[0].title == spec.title
    assert doc.slides[-1].title == "Questions?"
    for c, slide in zip(spec.concepts, doc.slides[1:-1], strict=True):
        assert slide.title == c.label
        assert slide.formula_hint == c.formula
        assert slide.code_hint == c.code
        assert slide.picture_count == int(c.picture)
        assert max((p.level for p in slide.paragraphs), default=0) == c.bullet_depth
        assert ("is defined as" in slide.text) == c.definition
        assert slide.text.count("For example,") == c.examples
        assert slide.text.count("Step ") == c.steps
        assert len(slide.text.split()) - len(c.label.split()) >= c.words


def test_pdf_round_trip() -> None:
    spec = make_deck_spec(3, 12)
    doc = parse_materials(build_pdf(spec), "x.pdf")
    assert doc.slide_count == len(spec.concepts) + 2
    for c, page in zip(spec.concepts, doc.slides[1:-1], strict=True):
        assert page.text.splitlines()[0] == c.label
        assert page.formula_hint == c.formula
        assert page.code_hint == c.code


def test_builders_are_deterministic_in_content() -> None:
    first = parse_materials(build_pptx(make_deck_spec(5)), "x.pptx")
    second = parse_materials(build_pptx(make_deck_spec(5)), "x.pptx")
    assert first == second
    assert build_pdf(make_deck_spec(5)) == build_pdf(make_deck_spec(5))
    assert build_c1_graph(make_deck_spec(5)) == build_c1_graph(make_deck_spec(5))
    assert build_labels_csv([make_deck_spec(5)]) == build_labels_csv([make_deck_spec(5)])


def test_plain_concept_has_no_stray_hints() -> None:
    deck = make_deck_spec(1, 1)
    deck.concepts = [_concept(definition=True, examples=2, steps=3, words=80, bullet_depth=3)]
    slide = parse_materials(build_pptx(deck), "x.pptx").slides[1]
    assert not detect_formula(slide.text)
    assert not detect_code(slide.text)


@pytest.mark.parametrize("version", ["v0", "v1"])
def test_graph_shape(version: str) -> None:
    spec = make_deck_spec(4, 9)
    response = C1GraphResponse.model_validate(build_c1_graph(spec, version))  # type: ignore[arg-type]
    ids = {n.id for n in response.graph.nodes}
    expected = {c.slug if version == "v1" else c.concept_id for c in spec.concepts}
    assert ids == expected
    assert response.graph.node_count == len(ids)
    assert response.concept_count == len(ids)
    assert response.graph.edge_count == len(response.graph.edges) > 0
    for edge in response.graph.edges:
        assert edge.source in ids
        assert edge.target in ids
        assert edge.evidence[0].page
        if edge.relation_type == "co_mentioned":
            assert edge.source < edge.target
    relations = {e.relation_type for e in response.graph.edges}
    prereqs = [e for e in response.graph.edges if e.relation_type == "prerequisite"]
    if version == "v0":
        assert relations == {"co_mentioned"}
        assert all(n.description is None for n in response.graph.nodes)
    else:
        assert relations == {"co_mentioned", "prerequisite"}
        assert len(prereqs) == sum(len(c.prerequisites) for c in spec.concepts)
        assert all(n.description for n in response.graph.nodes)
        assert {e.source_type for e in prereqs} == {"curriculum_defined", "inferred"}
        for e in prereqs:
            assert e.direction == "source_is_prerequisite_of_target"
            assert e.confidence is not None
            assert 0 <= e.confidence <= 1


@pytest.mark.parametrize(
    ("overrides", "n_prereq", "expected"),
    [
        ({"words": 10, "definition": True, "examples": 3, "picture": True}, 0, "Low"),
        ({"words": 50}, 1, "Medium"),
        ({"words": 100, "formula": True, "code": True}, 3, "High"),
    ],
)
def test_label_rule(overrides: dict, n_prereq: int, expected: str) -> None:
    assert synthetic_label(_concept(**overrides), n_prereq) == DifficultyLabel(expected)


def test_label_rule_covers_all_classes() -> None:
    labels = {
        synthetic_label(c, len(c.prerequisites))
        for s in range(5)
        for c in make_deck_spec(s, 10).concepts
    }
    assert labels == set(DifficultyLabel)


def test_labels_csv() -> None:
    specs = [make_deck_spec(1, 4), make_deck_spec(2, 4)]
    rows = list(csv.reader(io.StringIO(build_labels_csv(specs))))
    assert rows[0] == [
        "lecture_id",
        "concept_id",
        "concept_slug",
        "concept_label",
        "difficulty_label",
        "synthetic",
    ]
    assert len(rows) == 9
    assert {r[5] for r in rows[1:]} == {"true"}
    assert {r[4] for r in rows[1:]} <= {"Low", "Medium", "High"}
    assert all(r[2] == r[2].lower() and " " not in r[2] for r in rows[1:])
