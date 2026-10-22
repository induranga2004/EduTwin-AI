"""Lexical difficulty estimate.

Transparent heuristic baseline, NOT a trained model: it blends average word length,
share of long words (a cheap proxy for rare/technical vocabulary) and average sentence
length into a score in [0, 1].
"""

import re

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
_SRC = "materials:regex"
_VOCAB_SRC = "mock:todo_domain_vocabulary"


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


def language_features(concept: ConceptRecord, slides: list[SlideRecord]) -> list[FeatureValue]:
    """Group A features over the text of the concept's slides, in registry order."""
    unavail, done = FeatureStatus.UNAVAILABLE, FeatureStatus.COMPUTED
    names = features_of(FeatureGroup.LANGUAGE)
    if not slides:
        return [_fv(n, None, unavail, _SRC, _NO_SLIDES) for n in names]
    text = "\n".join(s.text for s in slides)
    words = _WORD_RE.findall(text)
    mock = _fv(
        "lang.unfamiliar_vocab_ratio",
        0.0,
        FeatureStatus.MOCKED,
        _VOCAB_SRC,
        "TODO: placeholder 0.0, no domain vocabulary list yet",
    )
    if not words:
        note = "no words on the concept's slides"
        return [_fv(n, None, unavail, _SRC, note) for n in names[:-1]] + [mock]
    n_words = len(words)
    units = [x for x in _UNIT_SPLIT_RE.split(text) if _WORD_RE.search(x)]
    long_share = sum(len(w) >= LONG_WORD_LEN for w in words) / n_words
    label = concept.label.strip()
    mentions = 0
    if label:
        mentions = len(re.findall(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text, re.IGNORECASE))
    return [
        _fv("lang.avg_sentence_length", n_words / max(1, len(units)), done, _SRC),
        _fv("lang.avg_word_length", sum(len(w) for w in words) / n_words, done, _SRC),
        _fv("lang.long_word_ratio", long_share, done, _SRC),
        _fv("lang.lexical_score", lexical_difficulty(text), done, _SRC),
        _fv("lang.concept_term_density", 100.0 * mentions / n_words, done, _SRC),
        mock,
    ]
