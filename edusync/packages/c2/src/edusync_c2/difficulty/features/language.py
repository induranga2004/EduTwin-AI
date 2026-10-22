"""Lexical difficulty estimate.

Transparent heuristic baseline, NOT a trained model: it blends average word length,
share of long words (a cheap proxy for rare/technical vocabulary) and average sentence
length into a score in [0, 1].
"""

import re
from collections.abc import Iterable

from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer, get_default_analyzer
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    FeatureValue,
    SlideRecord,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_SENTENCE_RE = re.compile(r"[.!?]+")
LONG_WORD_LEN = 9


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def lexical_difficulty(text: str) -> float:
    """Return a difficulty score in [0, 1]; 0.0 for text without words.

    score = 0.4 * word-length term + 0.3 * long-word share + 0.3 * sentence-length term,
    where the word-length term maps avg length 3..8 to 0..1 and the sentence-length term
    maps avg 5..30 words per sentence to 0..1.
    """
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    avg_len = sum(len(w) for w in words) / len(words)
    long_share = sum(len(w) >= LONG_WORD_LEN for w in words) / len(words)
    sentences = [s for s in _SENTENCE_RE.split(text) if _WORD_RE.search(s)]
    avg_sentence = len(words) / max(1, len(sentences))
    score = (
        0.4 * _clamp((avg_len - 3) / 5)
        + 0.3 * _clamp(long_share * 2)
        + 0.3 * _clamp((avg_sentence - 5) / 25)
    )
    return _clamp(score)


# ---------------------------------------------------------------- group A feature processor
_UNIT_SPLIT_RE = re.compile(r"[.!?\n]+")
_NO_SLIDES = "concept not found in materials"
_NO_WORDS = "no words on the concept's slides"
_NO_RESOURCES = "vocabulary resources unavailable"
_NO_VOCAB_WORDS = "no words for vocabulary analysis"
_SRC = "materials:regex"
_VOCAB_SOURCES = {
    "lang.unfamiliar_vocab_ratio": "materials:wordpiece",
    "lang.subword_fragmentation": "materials:wordpiece",
    "lang.domain_term_ratio": "materials:lexicon",
}


def _fv(
    name: str, value: float | None, status: FeatureStatus, source: str, note: str = ""
) -> FeatureValue:
    return FeatureValue(
        name=name,
        group=FeatureGroup.LANGUAGE,
        value=value,
        status=status,
        source=source,
        note=note,
    )


def _unavailable(names: Iterable[str], note: str) -> list[FeatureValue]:
    return [
        _fv(n, None, FeatureStatus.UNAVAILABLE, _VOCAB_SOURCES.get(n, _SRC), note) for n in names
    ]


def _vocab_features(text: str, vocab: VocabularyAnalyzer | None) -> list[FeatureValue]:
    analyzer = vocab if vocab is not None else get_default_analyzer()
    if analyzer is None:
        return _unavailable(_VOCAB_SOURCES, _NO_RESOURCES)
    stats = analyzer.analyze(text)
    if stats is None:
        return _unavailable(_VOCAB_SOURCES, _NO_VOCAB_WORDS)
    done = FeatureStatus.COMPUTED
    return [
        _fv(
            "lang.unfamiliar_vocab_ratio",
            stats.unfamiliar_share,
            done,
            _VOCAB_SOURCES["lang.unfamiliar_vocab_ratio"],
        ),
        _fv(
            "lang.subword_fragmentation",
            stats.fragmentation,
            done,
            _VOCAB_SOURCES["lang.subword_fragmentation"],
        ),
        _fv(
            "lang.domain_term_ratio",
            stats.domain_share,
            done,
            _VOCAB_SOURCES["lang.domain_term_ratio"],
        ),
    ]


def _count_mentions(text: str, labels: list[str]) -> int:
    """Mentions of any label, each occurrence once (distinct terms, longest first)."""
    terms = sorted({" ".join(t.split()) for t in labels if t.strip()}, key=lambda t: (-len(t), t))
    if not terms:
        return 0
    body = "|".join(r"\s+".join(re.escape(w) for w in t.split()) for t in terms)
    return len(re.findall(r"(?<!\w)(?:" + body + r")(?!\w)", text, re.IGNORECASE))


def language_features(
    concept: ConceptRecord, slides: list[SlideRecord], vocab: VocabularyAnalyzer | None = None
) -> list[FeatureValue]:
    """Group A features over the text of the concept's slides, in registry order.

    `vocab` is the analyzer for the three vocabulary features; None means the shared default
    analyzer, and if its resources cannot be loaded those features are unavailable.
    """
    done = FeatureStatus.COMPUTED
    names = features_of(FeatureGroup.LANGUAGE)
    if not slides:
        return _unavailable(names, _NO_SLIDES)
    text = "\n".join(s.text for s in slides)
    words = _WORD_RE.findall(text)
    if not words:
        return _unavailable(names, _NO_WORDS)
    n_words = len(words)
    units = [x for x in _UNIT_SPLIT_RE.split(text) if _WORD_RE.search(x)]
    long_share = sum(len(w) >= LONG_WORD_LEN for w in words) / n_words
    label = concept.label.strip()
    mentions = 0
    if concept.alias_labels:
        mentions = _count_mentions(text, [label, *concept.alias_labels])
    elif label:
        mentions = len(re.findall(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text, re.IGNORECASE))
    produced = {
        f.name: f
        for f in [
            _fv("lang.avg_sentence_length", n_words / max(1, len(units)), done, _SRC),
            _fv("lang.avg_word_length", sum(len(w) for w in words) / n_words, done, _SRC),
            _fv("lang.long_word_ratio", long_share, done, _SRC),
            _fv("lang.lexical_score", lexical_difficulty(text), done, _SRC),
            _fv("lang.concept_term_density", 100.0 * mentions / n_words, done, _SRC),
            *_vocab_features(text, vocab),
        ]
    }
    missing = [n for n in names if n not in produced]
    if missing:
        raise ValueError(f"language features not produced: {', '.join(missing)}")
    return [produced[n] for n in names]
