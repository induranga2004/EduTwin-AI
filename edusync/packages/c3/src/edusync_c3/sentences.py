"""Sentence splitting for lecture transcripts (no external dependencies).

Splits after ``.``, ``!`` or ``?`` followed by whitespace, except after common abbreviations,
initials ("J.K. Rowling"), and "No." or "Rs." before a number ("Act No. 36 of 2003", "Rs. 100,000").
A short line holding an equation without end punctuation ("SLE = Asset Value x Exposure Factor")
is kept as a sentence of its own instead of being glued to the next line.
"""

import re

_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "fig",
        "eq", "sec", "ch", "vol", "pp", "approx", "inc", "ltd", "co", "u.s", "u.k",
    }
)  # fmt: skip
_BEFORE_NUMBER = frozenset({"no", "rs"})  # only continue a sentence when a digit follows
_MAX_FORMULA_WORDS = 8


def _is_false_boundary(before: str, after: str) -> bool:
    """True when the period at the end of ``before`` does not end a sentence."""
    last = before.rsplit(None, 1)[-1] if before.split() else ""
    word = last.rstrip(".!?\"')]").lower()
    if not last.endswith("."):
        return False
    if word in _ABBREVIATIONS:
        return True
    if re.fullmatch(r"(?:[a-z]\.)*[a-z]", word):
        return True  # initials such as "J.K." or a single letter
    return word in _BEFORE_NUMBER and re.match(r"\d", after) is not None  # "No. 36", "Rs. 100,000"


def _is_formula_line(line: str) -> bool:
    """A short equation line with no end punctuation, e.g. ``ALE = SLE x ARO``."""
    return (
        "=" in line
        and not line[:1].islower()
        and not line.endswith((".", "!", "?"))
        and len(line.split()) <= _MAX_FORMULA_WORDS
    )


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into trimmed, non-empty sentences."""
    sentences: list[str] = []
    prose: list[str] = []
    for line in text.splitlines():
        if _is_formula_line(line.strip()):
            sentences += _split_prose(" ".join(prose))
            prose = []
            sentences.append(re.sub(r"\s+", " ", line).strip())
        else:
            prose.append(line)
    sentences += _split_prose(" ".join(prose))
    return [s for s in sentences if s]


def _split_prose(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        before, after = text[start : match.start()], text[match.end() :]
        if _is_false_boundary(before, after):
            continue
        sentences.append(text[start : match.start()].strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]
