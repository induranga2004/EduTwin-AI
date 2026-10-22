"""Concept building, slide location and the concept filter protocol."""

import pytest

from edusync_c2.difficulty.concepts import (
    ConceptFilter,
    FilterContext,
    MockConceptFilter,
    build_concepts,
    create_concept_filter,
    locate_slides,
    normalize,
)
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    C1GraphResponse,
    ConceptRecord,
    GraphContext,
    InputSource,
    MaterialsDoc,
)


def _ctx(resp: C1GraphResponse) -> GraphContext:
    return GraphContext(response=resp, source=InputSource(name="c1_graph", kind="real"))


def test_normalize() -> None:
    assert normalize("  Gradient \t DESCENT\n") == "gradient descent"


def test_locate_slides_word_boundaries(make_slide) -> None:
    slides = [
        make_slide(1, "Unlearning rates are different"),
        make_slide(2, "The Learning   Rate matters"),
        make_slide(3, "x", title="Learning rate"),
        make_slide(4, "learning rates plural"),
    ]
    assert locate_slides("learning rate", "learning rate", slides) == [2, 3]
    assert locate_slides("", "", slides) == []


def test_locate_slides_escapes_regex(make_slide) -> None:
    slides = [make_slide(1, "We use C++ here"), make_slide(2, "We use C here")]
    assert locate_slides("C++", "c++", slides) == [1]


def test_build_concepts_v1(deck: MaterialsDoc, graph_v1: C1GraphResponse) -> None:
    concepts = build_concepts(_ctx(graph_v1), deck)
    assert [c.id for c in concepts] == [
        "gradient descent",
        "learning rate",
        "loss function",
        "backpropagation",
    ]
    gd = concepts[0]
    assert gd.slide_indexes == [1, 2]
    assert gd.alignment_status is AlignmentStatus.ALIGNED
    assert gd.description == "About Gradient Descent."
    assert gd.frequency == 6
    assert gd.filter_status == "kept"
    assert gd.filter_note.startswith("mock: no filtering")


def test_unaligned_concept(deck: MaterialsDoc) -> None:
    resp = C1GraphResponse.model_validate(
        {"graph": {"nodes": [{"id": "quantum tunnelling", "label": "Quantum Tunnelling"}]}}
    )
    (c,) = build_concepts(_ctx(resp), deck)
    assert c.alignment_status is AlignmentStatus.NOT_FOUND
    assert c.slide_indexes == []


def test_duplicate_ids_keep_first(deck: MaterialsDoc) -> None:
    resp = C1GraphResponse.model_validate(
        {
            "graph": {
                "nodes": [
                    {"id": "loss function", "label": "Loss Function", "frequency": 5},
                    {"id": "loss function", "label": "Loss Function again", "frequency": 9},
                ]
            }
        }
    )
    (c,) = build_concepts(_ctx(resp), deck)
    assert c.frequency == 5


def test_fallback_to_concepts_list(deck: MaterialsDoc) -> None:
    resp = C1GraphResponse.model_validate(
        {"concepts": [{"name": "Learning  Rate", "frequency": 2}], "graph": {}}
    )
    (c,) = build_concepts(_ctx(resp), deck)
    assert (c.id, c.label, c.frequency, c.slide_indexes) == (
        "learning rate",
        "Learning  Rate",
        2,
        [2],
    )


class _DropFirst:
    name = "drop_first"

    def apply(self, concepts: list[ConceptRecord]) -> list[ConceptRecord]:
        return concepts[1:]


def test_custom_filter_and_protocol(deck: MaterialsDoc, graph_v0: C1GraphResponse) -> None:
    flt: ConceptFilter = _DropFirst()
    assert len(build_concepts(_ctx(graph_v0), deck, flt)) == 3
    assert MockConceptFilter.name == "mock_keep_all"
    assert len(build_concepts(_ctx(graph_v0), deck, MockConceptFilter())) == 4


def test_mock_flag_and_factory() -> None:
    assert MockConceptFilter.is_mock is True
    assert isinstance(create_concept_filter("mock_keep_all"), MockConceptFilter)
    assert isinstance(create_concept_filter("mock_keep_all", True), MockConceptFilter)
    with pytest.raises(NotImplementedError, match="heuristic"):
        create_concept_filter("heuristic")
    with pytest.raises(ValueError, match="unknown concept filter"):
        create_concept_filter("nope")


class _Merger:
    """Contextual filter: merges the first two records into the first and drops the last."""

    name = "fake_contextual"
    is_mock = False

    def __init__(self) -> None:
        self.seen: list[tuple[list[ConceptRecord], FilterContext]] = []

    def resolve(self, concepts: list[ConceptRecord], ctx: FilterContext) -> list[ConceptRecord]:
        self.seen.append((concepts, ctx))
        a, b, *rest = concepts
        merged = a.model_copy(
            update={
                "merged_from": [b.id],
                "alias_labels": [b.label],
                "slide_indexes": sorted({*a.slide_indexes, *b.slide_indexes}),
                "filter_note": "merged",
            }
        )
        dropped = rest[-1].model_copy(update={"filter_status": "dropped", "filter_note": "x"})
        return [merged, *rest[:-1], dropped]


def test_contextual_filter_dispatch(deck: MaterialsDoc, graph_v0: C1GraphResponse) -> None:
    flt = _Merger()
    ctx = _ctx(graph_v0)
    out = build_concepts(ctx, deck, flt)
    (concepts, fctx) = flt.seen[0]
    assert len(concepts) == 4
    assert fctx == FilterContext(graph=graph_v0, slides=tuple(deck.slides), graph_source=ctx.source)
    assert fctx.graph is ctx.response
    assert isinstance(fctx.slides, tuple)
    assert len(out) == 3
    assert out[0].merged_from == [concepts[1].id]
    assert out[0].alias_labels == [concepts[1].label]
    assert out[-1].filter_status == "dropped"
    with pytest.raises(AttributeError):
        fctx.slides = ()


def test_concept_record_merge_fields_default_empty() -> None:
    (c, *_) = build_concepts(
        _ctx(
            C1GraphResponse.model_validate(
                {"concepts": [{"name": "A", "frequency": 1}], "graph": {}}
            )
        ),
        MaterialsDoc(
            filename="a.pdf", file_type="pdf", slide_count=0, structure_available=False, slides=[]
        ),
    )
    assert (c.merged_from, c.alias_labels) == ([], [])
