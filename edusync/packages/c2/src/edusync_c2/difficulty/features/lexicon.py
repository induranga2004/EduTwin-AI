"""Domain-term lexicon for the language feature group (technical CS/IT/ML vocabulary)."""

import functools
from importlib.resources import files

_PACKAGE = "edusync_c2.difficulty"
_RESOURCE = ("resources", "domain_terms.txt")


class LexiconError(RuntimeError):
    """The domain lexicon is missing, unreadable or empty."""


def parse_lexicon(text: str) -> frozenset[str]:
    """Parse lexicon text: one term per line, `#` comments and blank lines skipped.

    Entries must already be lowercase; no case folding is applied here.
    """
    terms = frozenset(
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )
    if not terms:
        raise LexiconError("domain lexicon contains no terms")
    return terms


@functools.lru_cache(maxsize=1)
def load_lexicon() -> frozenset[str]:
    """Load the packaged lexicon once per process.

    Failures are not cached (lru_cache does not cache exceptions), so a later call retries.
    """
    try:
        text = files(_PACKAGE).joinpath(*_RESOURCE).read_text(encoding="utf-8")
    except OSError as exc:
        raise LexiconError(f"cannot read domain lexicon: {exc}") from exc
    return parse_lexicon(text)


def in_lexicon(word: str, lexicon: frozenset[str]) -> bool:
    """True if `word` (already lowercased) is a lexicon term or its simple plural."""
    if word in lexicon:
        return True
    if word.endswith("es") and word[:-2] in lexicon:
        return True
    return word.endswith("s") and not word.endswith("ss") and word[:-1] in lexicon
