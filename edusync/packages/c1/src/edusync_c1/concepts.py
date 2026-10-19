"""Heuristic concept extraction from lecture text (a transparent baseline, not a trained model)."""

import re
from collections import Counter

from edusync_c1.schemas import Concept

# Common words that are not useful concepts when extracted as standalone phrases.
STOP_WORDS = frozenset(
    {
        "the", "this", "that", "these", "those", "with", "from", "into", "when",
        "where", "which", "what", "each", "such", "more", "most", "other", "some",
        "also", "should", "would", "could", "have", "been", "being", "using", "used",
        "and", "for", "are", "was", "were", "has", "had", "will", "can", "may",
        "not", "but", "all", "any", "our", "your", "their", "its", "then", "than",
        "there", "here", "they", "them", "you", "we", "it", "is", "in", "on", "at",
        "to", "of", "by", "as", "an", "a", "or", "be", "do", "does",
    }
)  # fmt: skip

_PATTERNS = tuple(
    re.compile(p)
    for p in (
        # Capitalised phrases, e.g. "Binary Search Tree"
        r"\b[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*"
        r"(?:\s+[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*){0,3}\b",
        # Acronyms, e.g. "TCP-IP"
        r"\b[A-Z]{2,}(?:[- ][A-Z0-9]{2,}){0,2}\b",
        # Lower-case multi-word phrases, e.g. "gradient descent"
        r"\b[a-zA-Z][a-zA-Z0-9-]*(?:\s+[a-zA-Z][a-zA-Z0-9-]*){1,2}\b",
    )
)


def extract_concepts(text: str, max_concepts: int = 30) -> list[Concept]:
    """Extract candidate concepts, ranked by frequency with case-insensitive dedup."""
    if not text or not text.strip() or max_concepts <= 0:
        return []

    cleaned = re.sub(r"\s+", " ", text)
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for pattern in _PATTERNS:
        for match in pattern.finditer(cleaned):
            phrase = match.group(0).strip(" -")
            words = phrase.split()
            if len(phrase) < 3 or len(words) > 4:
                continue
            if all(word.casefold() in STOP_WORDS for word in words):
                continue
            key = " ".join(words).casefold()
            counts[key] += 1
            display.setdefault(key, phrase)

    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [Concept(name=display[key], frequency=counts[key]) for key in ranked[:max_concepts]]
