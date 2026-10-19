import re
from collections import Counter

# Common words that are not useful concepts when extracted as standalone phrases.
STOP_WORDS = {
    "the", "this", "that", "these", "those", "with", "from", "into", "when",
    "where", "which", "what", "each", "such", "more", "most", "other", "some",
    "also", "should", "would", "could", "have", "been", "being", "using", "used",
    "and", "for", "are", "was", "were", "has", "had", "will", "can", "may",
    "not", "but", "all", "any", "our", "your", "their", "its", "then", "than",
    "there", "here", "they", "them", "you", "we", "it", "is", "in", "on", "at",
    "to", "of", "by", "as", "an", "a", "or", "be", "do", "does",
}


def extract_concepts(text: str, max_concepts: int = 30) -> list[dict]:
    """
    Extract candidate technical concepts from arbitrary lecture text.

    This is a transparent heuristic baseline, not a trained model. Candidate
    phrases are ranked by frequency, with case-insensitive deduplication.
    """
    if not text or not text.strip() or max_concepts <= 0:
        return []

    cleaned = re.sub(r"\s+", " ", text)
    patterns = (
        r"\b[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*(?:\s+[A-Z][a-zA-Z0-9]+(?:[-'][A-Za-z0-9]+)*){0,3}\b",
        r"\b[A-Z]{2,}(?:[- ][A-Z0-9]{2,}){0,2}\b",
        r"\b[a-zA-Z][a-zA-Z0-9-]*(?:\s+[a-zA-Z][a-zA-Z0-9-]*){1,2}\b",
    )

    counts = Counter()
    display = {}
    for pattern in patterns:
        for match in re.finditer(pattern, cleaned):
            phrase = match.group(0).strip(" -")
            words = phrase.split()
            key = " ".join(words).casefold()
            if len(phrase) < 3 or len(words) > 4:
                continue
            if all(word.casefold() in STOP_WORDS for word in words):
                continue
            # Avoid phrases made entirely of generic stop words.
            if words[0].casefold() in STOP_WORDS and len(words) == 1:
                continue
            counts[key] += 1
            display.setdefault(key, phrase)

    ranked = sorted(counts, key=lambda key: (-counts[key], key))
    return [
        {"name": display[key], "frequency": int(counts[key])}
        for key in ranked[:max_concepts]
    ]
