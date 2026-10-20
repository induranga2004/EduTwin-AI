"""Concept building, slide location and the concept filter protocol."""

from edusync_c2.difficulty.concepts import (
    ConceptFilter,
    MockConceptFilter,
    build_concepts,
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
