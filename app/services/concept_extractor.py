
import re
from collections import Counter


def extract_concepts(text: str, max_concepts: int = 30) -> list[dict]:
    """
    Extract candidate concepts from lecture text.

    This is an initial baseline, not a trained research model.
    It identifies repeated technical terms and noun-like phrases.
    """
    if not text or not text.strip():
        return []

    cleaned = re.sub(r"\s+", " ", text)

    # Candidate phrases: capitalised terms and short multi-word phrases.
    patterns = [
        r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3}\b",
        r"\b[a-zA-Z][a-zA-Z-]+(?:\s+[a-zA-Z][a-zA-Z-]+){1,2}\b",
        r"\b[A-Z]{2,}(?:\s+[A-Z]{2,}){0,2}\b",
    ]

    stop_phrases = {
        "the", "this", "that", "these", "those", "with", "from",
        "into", "when", "where", "which", "what", "each", "such",
        "more", "most", "other", "some", "also", "should", "would",
        "could", "have", "been", "being", "using", "used",
    }

    candidates = Counter()

    for pattern in patterns:
        for match in re.finditer(pattern, cleaned):
            phrase = match.group(0).strip(" -")
            words = phrase.split()
            normalised = phrase.lower()

            if len(phrase) < 3 or len(words) > 4:
                continue

            if normalised in stop_phrases:
                continue

            if all(word.lower() in stop_phrases for word in words):
                continue

            candidates[phrase] += 1

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-item[1], item[0].lower()),
    )

    results = []
    seen = set()

    for phrase, count in ranked:
        key = phrase.lower()

        if key in seen:
            continue

        seen.add(key)
        results.append(
            {
                "name": phrase,
                "frequency": count,
            }
        )

        if len(results) >= max_concepts:
            break

    return results
