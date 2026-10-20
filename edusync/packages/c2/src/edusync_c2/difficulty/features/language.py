"""Lexical difficulty estimate.

Transparent heuristic baseline, NOT a trained model: it blends average word length,
share of long words (a cheap proxy for rare/technical vocabulary) and average sentence
length into a score in [0, 1].
"""

import re

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
