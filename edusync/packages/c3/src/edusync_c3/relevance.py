"""Pedagogical relevance of speech via keyword overlap.

Transparent baseline, NOT a semantic model: it only measures how many learning-objective
terms appear in the text, so paraphrases without shared words score low.
"""

import re
from enum import StrEnum

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class RelevanceLabel(StrEnum):
    RELEVANT = "relevant"
    UNRELATED = "unrelated"
    UNCERTAIN = "uncertain"


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def keyword_overlap_score(text: str, concepts: list[str]) -> float:
    """Fraction of distinct concept terms found in ``text`` (0.0 with no concepts/terms)."""
    terms = set().union(*(_tokens(c) for c in concepts)) if concepts else set()
    if not terms:
        return 0.0
    return len(terms & _tokens(text)) / len(terms)


def classify(
    text: str,
    concepts: list[str],
    relevant_threshold: float = 0.2,
    unrelated_threshold: float = 0.05,
) -> RelevanceLabel:
    """Label text relevant (score >= relevant_threshold), unrelated (< unrelated_threshold)
    or uncertain (in between)."""
    score = keyword_overlap_score(text, concepts)
    if score >= relevant_threshold:
        return RelevanceLabel.RELEVANT
    if score < unrelated_threshold:
        return RelevanceLabel.UNRELATED
    return RelevanceLabel.UNCERTAIN
