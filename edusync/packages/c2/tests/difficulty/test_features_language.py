"""Group A: language features."""

import pytest

from edusync_c2.difficulty import lexical_difficulty
from edusync_c2.difficulty.features.language import language_features
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
)

NAMES = list(features_of(FeatureGroup.LANGUAGE))


def concept(label: str = "Gradient Descent") -> ConceptRecord:
    return ConceptRecord(
        id="gd", label=label, alignment_status=AlignmentStatus.ALIGNED, slide_indexes=[1]
    )


def by_name(fs):
    return {f.name: f for f in fs}


def test_registry_order_and_values(make_slide):
    slide = make_slide(1, "Gradient descent is great. Gradient descent converges.")
    fs = language_features(concept(), [slide])
    assert [f.name for f in fs] == NAMES
    d = by_name(fs)
    assert d["lang.avg_sentence_length"].value == pytest.approx(3.5)  # 7 words, 2 sentences
    assert d["lang.concept_term_density"].value == pytest.approx(100 * 2 / 7)
    assert d["lang.lexical_score"].value == lexical_difficulty(slide.text)
    assert d["lang.long_word_ratio"].value == pytest.approx(1 / 7)  # "converges" = 9 letters
    assert d["lang.avg_word_length"].value > 0
    assert d["lang.unfamiliar_vocab_ratio"].status is FeatureStatus.MOCKED
    assert d["lang.unfamiliar_vocab_ratio"].value == 0.0
    assert all(
        f.status is FeatureStatus.COMPUTED for f in fs if f.name != "lang.unfamiliar_vocab_ratio"
    )


def test_no_slides_unavailable():
    fs = language_features(concept(), [])
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.UNAVAILABLE and f.value is None for f in fs)
    assert all("not found" in f.note for f in fs)


def test_zero_words(make_slide):
    fs = language_features(concept(), [make_slide(1, "123 ... !!!")])
    assert [f.name for f in fs] == NAMES
    assert [f.status for f in fs[:-1]] == [FeatureStatus.UNAVAILABLE] * 5
    assert fs[-1].status is FeatureStatus.MOCKED


def test_empty_label_and_determinism(make_slide):
    slide = make_slide(1, "Some words here")
    a = language_features(concept(" "), [slide])
    assert by_name(a)["lang.concept_term_density"].value == 0.0
    assert a == language_features(concept(" "), [slide])
