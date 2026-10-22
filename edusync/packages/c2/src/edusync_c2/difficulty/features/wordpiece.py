"""Pure-Python BERT WordPiece (uncased) matching the MiniLM tokenizer, without special tokens."""

import unicodedata
from importlib import resources

UNK = "[UNK]"
MAX_WORD_CHARS = 100
_CONT = "##"
_CACHE_MAX = 100_000
_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B73F),
    (0x2B740, 0x2B81F),
    (0x2B820, 0x2CEAF),
    (0xF900, 0xFAFF),
    (0x2F800, 0x2FA1F),
)


def _is_space(ch: str) -> bool:
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def _clean(text: str) -> str:
    kept = (
        " " if _is_space(ch) else ch
        for ch in text
        if ch not in "\x00\ufffd" and not _is_control(ch)
    )
    return "".join(kept)


def _fold(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _split_punct(word: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    for ch in word:
        if _is_punct(ch):
            if cur:
                parts.append("".join(cur))
                cur = []
            parts.append(ch)
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


class WordPieceTokenizer:
    """Greedy longest-match-first WordPiece over an ordered vocabulary.

    Results per word are memoised in a per-instance dict. A dict get or set is atomic under the
    GIL and the cached values are immutable tuples of a pure function, so concurrent use is safe
    (a race only repeats work). Nothing else is mutable after construction.
    """

    def __init__(self, vocab: list[str]) -> None:
        self._vocab = {token: i for i, token in enumerate(vocab)}
        self._cache: dict[str, tuple[str, ...]] = {}

    @classmethod
    def from_resources(cls) -> "WordPieceTokenizer":
        """Build from the vendored bert_uncased_vocab.txt."""
        ref = resources.files("edusync_c2.difficulty").joinpath(
            "resources", "bert_uncased_vocab.txt"
        )
        entries = ref.read_bytes().decode("utf-8").split("\n")
        return cls(entries[:-1] if entries[-1] == "" else entries)

    def normalize(self, text: str) -> list[str]:
        """BERT basic tokenization: clean, pad CJK, lowercase, strip accents, split punctuation."""
        cleaned = "".join(f" {ch} " if _is_cjk(ch) else ch for ch in _clean(text))
        words: list[str] = []
        for chunk in cleaned.split():
            words.extend(_split_punct(_fold(chunk)))
        return words

    def pieces(self, word: str) -> tuple[str, ...]:
        """WordPiece segmentation of one normalised word; (UNK,) when it cannot be segmented."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        result = self._segment(word)
        if len(self._cache) < _CACHE_MAX:
            self._cache[word] = result
        return result

    def tokenize(self, text: str) -> list[str]:
        """Pieces of every word of the text, without special tokens."""
        out: list[str] = []
        for word in self.normalize(text):
            out.extend(self.pieces(word))
        return out

    def _segment(self, word: str) -> tuple[str, ...]:
        if len(word) > MAX_WORD_CHARS:
            return (UNK,)
        found: list[str] = []
        start = 0
        while start < len(word):
            end = len(word)
            match = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = _CONT + sub
                if sub in self._vocab:
                    match = sub
                    break
                end -= 1
            if match is None:
                return (UNK,)
            found.append(match)
            start = end
        return tuple(found)
