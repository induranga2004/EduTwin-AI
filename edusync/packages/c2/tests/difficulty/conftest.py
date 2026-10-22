"""Shared fixtures for the difficulty tests: a hand-built deck and C1 graphs (v0 and v1)."""

import random
from collections.abc import Callable
from typing import Any

import pytest

from edusync_c2.difficulty.features.vocab import reset_default_analyzer
from edusync_c2.difficulty.schemas import (
    C1GraphResponse,
    MaterialsDoc,
    ParagraphRecord,
    SlideRecord,
)

SlideFactory = Callable[..., SlideRecord]
VariantFactory = Callable[..., dict[str, list[SlideRecord]]]

UNRELATED_SLIDE_TEXT = "Cryptography encryption decryption malleability nonce ciphertext."


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
        **geometry: Any,
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
            **geometry,
        )

    return build


@pytest.fixture
def slide_variants(make_slide: SlideFactory) -> VariantFactory:
    """Metamorphic variants of a slide list, for invariants a feature must keep.

    Keys: ``uppercase`` (all text upper-cased), ``duplicated_slides`` (the whole list twice),
    ``doubled_text`` (each slide's text repeated inside the same slide), ``unrelated_extra`` (one
    more slide about something else, appended last) and ``shuffled_1`` .. ``shuffled_<n>`` (the
    same slides in a seeded random order, indexes unchanged).
    """

    def build(
        slides: list[SlideRecord], *, shuffles: int = 5, seed: int = 7
    ) -> dict[str, list[SlideRecord]]:
        last = max(s.index for s in slides)
        variants = {
            "uppercase": [
                s.model_copy(
                    update={
                        "text": s.text.upper(),
                        "paragraphs": [
                            p.model_copy(update={"text": p.text.upper()}) for p in s.paragraphs
                        ],
                    }
                )
                for s in slides
            ],
            "duplicated_slides": [
                *slides,
                *(s.model_copy(update={"index": last + s.index}) for s in slides),
            ],
            "doubled_text": [
                s.model_copy(
                    update={
                        "text": s.text + "\n" + s.text,
                        "paragraphs": [*s.paragraphs, *s.paragraphs],
                    }
                )
                for s in slides
            ],
            "unrelated_extra": [*slides, make_slide(last + 1, UNRELATED_SLIDE_TEXT)],
        }
        rng = random.Random(seed)
        for n in range(1, shuffles + 1):
            shuffled = slides[:]
            rng.shuffle(shuffled)
            variants[f"shuffled_{n}"] = shuffled
        return variants

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


class KeywordEncoder:
    """Fake sentence encoder: prototypes sit on one axis each; DEFN / EXMP mark test sentences.

    A text containing DEFN has cosine 0.9 to the definition prototypes (and 0.1 to the example
    ones); EXMP is the mirror image; any other text is orthogonal to all prototypes.
    """

    dims = 4
    sha = "abcd1234"

    def __init__(self) -> None:
        from edusync_c2.difficulty.features.semantic import CLASSES, load_prototypes

        protos = load_prototypes()
        self.axis = {t: i for i, name in enumerate(CLASSES) for t in protos[name]}
        self.calls = 0

    def encode(self, texts):
        import numpy as np

        self.calls += 1
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            if text in self.axis:
                out[row, self.axis[text]] = 1.0
            elif "DEFN" in text:
                out[row] = (0.9, 0.1, 0.0, 0.18**0.5)
            elif "EXMP" in text:
                out[row] = (0.1, 0.9, 0.0, 0.18**0.5)
            else:
                out[row, 3] = 1.0
        return out


@pytest.fixture
def keyword_encoder() -> KeywordEncoder:
    return KeywordEncoder()


@pytest.fixture
def make_scorer(keyword_encoder: KeywordEncoder):
    """Factory of a SemanticScorer over the keyword encoder (thresholds as given)."""

    def build(**kwargs: Any):
        from edusync_c2.difficulty.features.semantic import SemanticScorer

        return SemanticScorer(keyword_encoder, **kwargs)

    return build
