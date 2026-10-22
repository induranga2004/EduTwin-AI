"""Group A: language features."""

from typing import ClassVar

import pytest

from edusync_c2.difficulty import lexical_difficulty
from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.features import language as language_module
from edusync_c2.difficulty.features.language import language_features
from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer
from edusync_c2.difficulty.registry import FEATURE_NAMES, features_of
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    InputSource,
    PrerequisiteMetrics,
)

pytestmark = pytest.mark.usefixtures("fresh_default_analyzer")

NAMES = list(features_of(FeatureGroup.LANGUAGE))
VOCAB_NAMES = [
    "lang.unfamiliar_vocab_ratio",
    "lang.subword_fragmentation",
    "lang.domain_term_ratio",
]
EXPECTED_NAMES = [
    "lang.avg_sentence_length",
    "lang.avg_word_length",
    "lang.long_word_ratio",
    "lang.lexical_score",
    "lang.concept_term_density",
    *VOCAB_NAMES,
]
TECH = (
    "Backpropagation computes the gradient of the loss. The Hessian eigenvalue governs "
    "hyperparameter sensitivity.\nThe kernel schedules the process on the cpu."
)


class FakeTokenizer:
    TABLE: ClassVar = {"gradient": ["grad", "##ient"], "converges": ["conv", "##erges"]}

    def tokenize(self, text: str) -> list[str]:
        return self.TABLE.get(text, [text])


def fake_analyzer() -> VocabularyAnalyzer:
    return VocabularyAnalyzer(FakeTokenizer(), frozenset({"gradient", "descent"}))  # type: ignore[arg-type]


def concept(label: str = "Gradient Descent") -> ConceptRecord:
    return ConceptRecord(
        id="gd", label=label, alignment_status=AlignmentStatus.ALIGNED, slide_indexes=[1]
    )


def by_name(fs):
    return {f.name: f for f in fs}


def vocab_values(fs):
    d = by_name(fs)
    return [d[n].value for n in VOCAB_NAMES]


def test_registry_order_is_the_exact_expected_list():
    assert NAMES == EXPECTED_NAMES


def test_registry_order_and_values(make_slide):
    slide = make_slide(1, "Gradient descent is great. Gradient descent converges.")
    fs = language_features(concept(), [slide], fake_analyzer())
    assert [f.name for f in fs] == NAMES
    d = by_name(fs)
    assert d["lang.avg_sentence_length"].value == pytest.approx(3.5)  # 7 words, 2 sentences
    assert d["lang.concept_term_density"].value == pytest.approx(100 * 2 / 7)
    assert d["lang.lexical_score"].value == lexical_difficulty(slide.text)
    assert d["lang.long_word_ratio"].value == pytest.approx(1 / 7)  # "converges" = 9 letters
    assert d["lang.avg_word_length"].value > 0
    # pieces: gradient 2, descent 1, is 1, great 1, gradient 2, descent 1, converges 2
    assert d["lang.subword_fragmentation"].value == pytest.approx(10 / 7)
    assert d["lang.unfamiliar_vocab_ratio"].value == pytest.approx(3 / 7)
    assert d["lang.domain_term_ratio"].value == pytest.approx(4 / 7)
    assert all(f.status is FeatureStatus.COMPUTED for f in fs)
    assert all(f.note == "" for f in fs)


def test_sources(make_slide):
    fs = by_name(language_features(concept(), [make_slide(1, "Some words here")], fake_analyzer()))
    assert {n: fs[n].source for n in NAMES} == {
        "lang.avg_sentence_length": "materials:regex",
        "lang.avg_word_length": "materials:regex",
        "lang.long_word_ratio": "materials:regex",
        "lang.lexical_score": "materials:regex",
        "lang.concept_term_density": "materials:regex",
        "lang.unfamiliar_vocab_ratio": "materials:wordpiece",
        "lang.subword_fragmentation": "materials:wordpiece",
        "lang.domain_term_ratio": "materials:lexicon",
    }


def test_real_default_analyzer_smoke(make_slide):
    fs = language_features(concept(), [make_slide(1, TECH)])
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.COMPUTED for f in fs)
    frag, unfamiliar, domain = (
        by_name(fs)["lang.subword_fragmentation"].value,
        by_name(fs)["lang.unfamiliar_vocab_ratio"].value,
        by_name(fs)["lang.domain_term_ratio"].value,
    )
    assert frag > 1.0
    assert 0.0 < unfamiliar < 1.0
    assert 0.0 < domain < 1.0


def test_no_slides_unavailable():
    fs = language_features(concept(), [])
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.UNAVAILABLE and f.value is None for f in fs)
    assert all("not found" in f.note for f in fs)


def test_zero_words_all_unavailable(make_slide):
    class Untouched:
        def analyze(self, text):
            raise AssertionError("must not be called")

    fs = language_features(concept(), [make_slide(1, "123 ... !!!")], Untouched())  # type: ignore[arg-type]
    assert [f.name for f in fs] == NAMES
    assert all(f.status is FeatureStatus.UNAVAILABLE and f.value is None for f in fs)
    assert all(f.note == "no words on the concept's slides" for f in fs)


def test_resources_unavailable_only_affects_vocab_features(make_slide, monkeypatch):
    monkeypatch.setattr(language_module, "get_default_analyzer", lambda: None)
    fs = language_features(concept(), [make_slide(1, "Gradient descent is great.")])
    assert [f.name for f in fs] == NAMES
    d = by_name(fs)
    for name in VOCAB_NAMES:
        assert d[name].status is FeatureStatus.UNAVAILABLE
        assert d[name].value is None
        assert d[name].note == "vocabulary resources unavailable"
    assert d["lang.unfamiliar_vocab_ratio"].status is not FeatureStatus.MOCKED
    assert [d[n].status for n in NAMES[:5]] == [FeatureStatus.COMPUTED] * 5


def test_vocab_features_unavailable_when_analyzer_finds_no_words(make_slide):
    fs = language_features(concept(), [make_slide(1, "A b c")], fake_analyzer())
    d = by_name(fs)
    for name in VOCAB_NAMES:
        assert d[name].status is FeatureStatus.UNAVAILABLE
        assert d[name].note == "no words for vocabulary analysis"
    assert [d[n].status for n in NAMES[:5]] == [FeatureStatus.COMPUTED] * 5


def test_explicit_vocab_never_touches_the_default(make_slide, monkeypatch):
    def boom():
        raise AssertionError("default analyzer must not be used")

    monkeypatch.setattr(language_module, "get_default_analyzer", boom)
    fs = language_features(concept(), [make_slide(1, "Gradient descent")], fake_analyzer())
    assert all(f.status is FeatureStatus.COMPUTED for f in fs)


def test_unproduced_registry_name_raises(make_slide, monkeypatch):
    monkeypatch.setattr(
        language_module, "features_of", lambda group: (*features_of(group), "lang.bogus")
    )
    with pytest.raises(ValueError, match=r"lang.bogus"):
        language_features(concept(), [make_slide(1, "Gradient descent")], fake_analyzer())


def test_empty_label_and_determinism(make_slide):
    slide = make_slide(1, "Some words here")
    a = language_features(concept(" "), [slide])
    assert by_name(a)["lang.concept_term_density"].value == 0.0
    assert a == language_features(concept(" "), [slide])


# ------------------------------------------------------------------ metamorphic properties
def test_uppercase_text_keeps_vocab_features(make_slide, slide_variants):
    slides = [make_slide(1, TECH)]
    lower = language_features(concept(), slides)
    upper = language_features(concept(), slide_variants(slides)["uppercase"])
    assert vocab_values(lower) == vocab_values(upper)


def test_duplicated_slide_text_keeps_ratios(make_slide, slide_variants):
    slides = [make_slide(1, TECH)]
    variants = slide_variants(slides)
    once = language_features(concept(), slides)
    twice = language_features(concept(), variants["duplicated_slides"])
    assert len(variants["duplicated_slides"]) == 2
    assert vocab_values(once) == vocab_values(twice)
    doubled = language_features(concept(), variants["doubled_text"])
    assert vocab_values(once) == vocab_values(doubled)


def test_slide_order_does_not_matter(make_slide, slide_variants):
    texts = [
        "Backpropagation computes the gradient.",
        "The kernel schedules the process on the cpu.",
        "We went to the park on a sunny day.",
        "Eigenvalue and hyperparameter tuning.",
    ]
    slides = [make_slide(i + 1, t) for i, t in enumerate(texts)]
    reference = vocab_values(language_features(concept(), slides))
    shuffled = {k: v for k, v in slide_variants(slides).items() if k.startswith("shuffled_")}
    assert len(shuffled) == 5
    assert any([s.index for s in v] != [1, 2, 3, 4] for v in shuffled.values())
    for variant in shuffled.values():
        assert vocab_values(language_features(concept(), variant)) == reference


def compute(concept_record, slides, **kw):
    return compute_all_features(
        concept_record,
        all_concepts=[concept_record],
        slides_by_index={s.index: s for s in slides},
        structure_available=True,
        metrics=PrerequisiteMetrics(
            direct_count=1,
            chain_depth=1,
            transitive_count=1,
            mean_confidence=0.5,
            curriculum_defined_share=1.0,
        ),
        graph_has_prerequisite_edges=True,
        graph_source=InputSource(name="c1_graph", kind="real"),
        **kw,
    )


def test_unrelated_slide_changes_nothing(make_slide, slide_variants):
    rec = ConceptRecord(
        id="gd",
        label="Gradient Descent",
        alignment_status=AlignmentStatus.ALIGNED,
        slide_indexes=[2, 1],
    )
    own = [make_slide(1, TECH), make_slide(2, "Gradient descent is great.")]
    extra = slide_variants(own)["unrelated_extra"]
    assert [s.index for s in extra] == [1, 2, 3]
    before = compute(rec, own)[:8]
    after = compute(rec, extra)[:8]
    assert before == after
    assert [f.name for f in before] == NAMES


def test_compute_all_features_passes_the_analyzer_through(make_slide):
    rec = concept()
    slides = [make_slide(1, "Gradient descent converges.")]
    fs = compute(rec, slides, vocab=fake_analyzer())
    assert len(fs) == len(FEATURE_NAMES)
    d = by_name(fs[:8])
    assert d["lang.domain_term_ratio"].value == pytest.approx(2 / 3)
    assert d["lang.subword_fragmentation"].value == pytest.approx(5 / 3)
