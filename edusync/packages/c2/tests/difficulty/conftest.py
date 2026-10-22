"""Shared fixtures for the difficulty tests: a hand-built deck and C1 graphs (v0 and v1)."""

from collections.abc import Callable

import pytest

from edusync_c2.difficulty.features.vocab import reset_default_analyzer
from edusync_c2.difficulty.schemas import (
    C1GraphResponse,
    MaterialsDoc,
    ParagraphRecord,
    SlideRecord,
)

SlideFactory = Callable[..., SlideRecord]


@pytest.fixture
def fresh_default_analyzer():
    """Forget the cached default vocabulary analyzer (or cached failure) around a test."""
    reset_default_analyzer()
    yield
    reset_default_analyzer()


@pytest.fixture
def make_slide() -> SlideFactory:
    def build(
        index: int,
        text: str,
        *,
        title: str | None = None,
        levels: list[int] | None = None,
        pictures: int | None = 0,
        charts: int | None = 0,
        tables: int | None = 0,
        shapes: int | None = 3,
        formula: bool = False,
        code: bool = False,
    ) -> SlideRecord:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        levels = levels or [0] * len(lines)
        paragraphs = [
            ParagraphRecord(text=ln, level=lv) for ln, lv in zip(lines, levels, strict=False)
        ]
        return SlideRecord(
            index=index,
            title=title,
            text=text,
            paragraphs=paragraphs,
            picture_count=pictures,
            chart_count=charts,
            table_count=tables,
            shape_count=shapes,
            formula_hint=formula,
            code_hint=code,
        )

    return build


@pytest.fixture
def deck(make_slide: SlideFactory) -> MaterialsDoc:
    """Five PPTX-like slides: definition + example, learning rate, formula, steps, no concept."""
    slides = [
        make_slide(
            1,
            "Gradient Descent\n"
            "Gradient descent is defined as an algorithm that minimises a loss function.\n"
            "For example, it updates weights after every pass.",
            title="Gradient Descent",
            levels=[0, 1, 1],
            pictures=1,
        ),
        make_slide(
            2,
            "Learning Rate\nThe learning rate controls the step size of gradient descent.\n"
            "A large learning rate can overshoot the minimum.",
            title="Learning Rate",
            levels=[0, 1, 2],
            shapes=4,
        ),
        make_slide(
            3,
            "Loss Function\nL(w) = sum((y - w*x)^2) / n",
            title="Loss Function",
            formula=True,
            charts=1,
        ),
        make_slide(
            4,
            "Backpropagation\nStep 1: run the forward pass.\n"
            "Step 2: compute gradients layer by layer.\n"
            "Step 3: update the weights.",
            title="Backpropagation",
            code=True,
        ),
        make_slide(5, "Questions?", title="Questions?", shapes=1),
    ]
    return MaterialsDoc(
        filename="lecture.pptx",
        file_type="pptx",
        slide_count=len(slides),
        structure_available=True,
        slides=slides,
    )


def _nodes(with_description: bool) -> list[dict]:
    base = [
        ("gradient descent", "Gradient Descent", 6),
        ("learning rate", "Learning Rate", 4),
        ("loss function", "Loss Function", 5),
        ("backpropagation", "Backpropagation", 3),
    ]
    return [
        {
            "id": i,
            "label": label,
            "frequency": freq,
            **({"description": f"About {label}."} if with_description else {}),
        }
        for i, label, freq in base
    ]


@pytest.fixture
def graph_v0() -> C1GraphResponse:
    """C1 sample shape today: co-mention edges only, no prerequisites."""
    edges = [
        {
            "source": "gradient descent",
            "target": "loss function",
            "relation_type": "co_mentioned",
            "evidence_count": 1,
            "evidence": [{"page": 1, "sentence": "Gradient descent minimises a loss function."}],
        }
    ]
    return C1GraphResponse.model_validate(
        {
            "lecture_id": "lec-0",
            "filename": "lecture.pptx",
            "page_count": 5,
            "concept_count": 4,
            "concepts": [{"name": n["label"], "frequency": n["frequency"]} for n in _nodes(False)],
            "graph": {
                "node_count": 4,
                "edge_count": 1,
                "nodes": _nodes(False),
                "edges": edges,
                "note": "co-mention",
            },
            "message": "Candidate knowledge graph generated successfully.",
        }
    )


def _prereq(source: str, target: str, confidence: float, source_type: str) -> dict:
    return {
        "source": source,
        "target": target,
        "relation_type": "prerequisite",
        "direction": "source_is_prerequisite_of_target",
        "confidence": confidence,
        "source_type": source_type,
        "evidence_count": 1,
        "evidence": [{"page": 1, "sentence": f"{source} comes before {target}."}],
    }


@pytest.fixture
def graph_v1() -> C1GraphResponse:
    """C1 contract v1: prerequisite edges with direction, confidence and source_type."""
    edges = [
        _prereq("loss function", "gradient descent", 0.9, "curriculum_defined"),
        _prereq("learning rate", "gradient descent", 0.6, "inferred"),
        _prereq("gradient descent", "backpropagation", 0.7, "inferred"),
        {"source": "learning rate", "target": "loss function", "relation_type": "co_mentioned"},
    ]
    return C1GraphResponse.model_validate(
        {
            "lecture_id": "lec-1",
            "filename": "lecture.pptx",
            "page_count": 5,
            "concept_count": 4,
            "concepts": [{"name": n["label"], "frequency": n["frequency"]} for n in _nodes(True)],
            "graph": {"node_count": 4, "edge_count": 4, "nodes": _nodes(True), "edges": edges},
        }
    )
