"""Vocabulary statistics for group A: word-piece fragmentation, unfamiliar and domain-term shares.

Words are runs of Unicode letters (hyphens, apostrophes, digits and underscores split words);
words shorter than two letters are ignored. Pure counting over a deterministic tokenizer, so the
result does not depend on hash seeds, thread timing or the order of the slides.
"""

import hashlib
import re
import threading
from dataclasses import dataclass
from importlib import resources

from edusync_c2.difficulty.features.lexicon import LexiconError, in_lexicon, load_lexicon
from edusync_c2.difficulty.features.wordpiece import UNK, WordPieceTokenizer
from edusync_c2.logger import get_logger

logger = get_logger(__name__)

MIN_PIECES_UNFAMILIAR = 2
MIN_WORD_LEN = 2
WORD_RE = re.compile(r"[^\W\d_]+")
_VOCAB_RESOURCE = ("resources", "bert_uncased_vocab.txt")


@dataclass(frozen=True)
class VocabStats:
    """Vocabulary measurements over one text; `words` is always at least 1."""

    words: int
    fragmentation: float
    unfamiliar_share: float
    domain_share: float


class VocabularyAnalyzer:
    """Word-piece tokenizer plus domain lexicon; immutable after construction, thread-safe."""

    def __init__(self, tokenizer: WordPieceTokenizer, lexicon: frozenset[str]) -> None:
        self.tokenizer = tokenizer
        self.lexicon = lexicon

    @classmethod
    def from_resources(cls) -> "VocabularyAnalyzer":
        """Build from the vendored MiniLM vocabulary and the curated domain lexicon."""
        return cls(WordPieceTokenizer.from_resources(), load_lexicon())

    def analyze(self, text: str) -> VocabStats | None:
        """Statistics over the words of `text`; None when it holds no words."""
        words = pieces_total = unfamiliar = domain = 0
        for match in WORD_RE.finditer(text):
            if len(match.group()) < MIN_WORD_LEN:
                continue
            word = match.group().lower()
            pieces = self.tokenizer.tokenize(word)
            if not pieces:
                continue
            words += 1
            pieces_total += len(pieces)
            if len(pieces) >= MIN_PIECES_UNFAMILIAR or UNK in pieces:
                unfamiliar += 1
            if in_lexicon(word, self.lexicon):
                domain += 1
        if not words:
            return None
        return VocabStats(
            words=words,
            fragmentation=pieces_total / words,
            unfamiliar_share=unfamiliar / words,
            domain_share=domain / words,
        )


class _Cache:
    """Process-wide result of the first load: an analyzer, or None when resources failed."""

    loaded = False
    analyzer: VocabularyAnalyzer | None = None


_cache = _Cache()
_lock = threading.Lock()


def _vocab_sha() -> str:
    ref = resources.files("edusync_c2.difficulty").joinpath(*_VOCAB_RESOURCE)
    return hashlib.sha256(ref.read_bytes()).hexdigest()[:8]


def _load() -> VocabularyAnalyzer | None:
    try:
        analyzer = VocabularyAnalyzer.from_resources()
        sha = _vocab_sha()
    except (LexiconError, OSError, ValueError) as exc:  # UnicodeDecodeError is a ValueError
        logger.warning("vocab_unavailable reason=%s", type(exc).__name__)
        return None
    logger.info(
        "vocab_loaded words=%d lexicon=%d sha=%s",
        analyzer.tokenizer.vocab_size,
        len(analyzer.lexicon),
        sha,
    )
    return analyzer


def get_default_analyzer() -> VocabularyAnalyzer | None:
    """The shared analyzer over the packaged resources; None (cached) if they cannot be loaded."""
    if not _cache.loaded:
        with _lock:
            if not _cache.loaded:
                _cache.analyzer = _load()
                _cache.loaded = True
    return _cache.analyzer


def reset_default_analyzer() -> None:
    """Forget the cached analyzer or failure so the next call loads again (for tests)."""
    with _lock:
        _cache.analyzer = None
        _cache.loaded = False
