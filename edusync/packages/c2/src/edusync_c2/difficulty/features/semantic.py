"""Prototype-based semantic scorer: how close is a slide sentence to a definition or an example?

A sentence is embedded with a sentence encoder (``encode(texts) -> (n, dims)`` unit vectors, see
``minilm.MiniLMEncoder``) and compared by cosine similarity with the centroid of a few hand-written
prototype sentences per class (``resources/semantic_prototypes.txt``: ``[definition]``,
``[example]`` and ``[step]``). numpy is imported lazily, so ``import edusync_c2`` never needs it.

Counting rule (``count_semantic``): a sentence counts toward a class when its cosine to that
class is at least the class threshold AND higher than its cosine to the other class; a tie counts
for neither, so no sentence is counted twice.

The thresholds (default 0.23 for both) are heuristic and NOT validated against human ratings. A
small earlier check (65 hand-written sentences) put useful thresholds around 0.21-0.25, with the
example class the weakest. They are settings, not facts.

Determinism: sentences are split and de-duplicated in input order, each text is encoded alone by
the encoder, and scores are rounded to 4 decimals. The scorer keeps a bounded embedding cache keyed
by sentence text (shared by the per-request views from ``for_request``), so a sentence repeated
across concepts is encoded once. ``for_request`` also applies the per-request sentence budget.
"""

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import TYPE_CHECKING, Any, Protocol

from edusync_c2.difficulty.errors import SemanticScoringError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[Any]

PROTOTYPE_RESOURCE = ("resources", "semantic_prototypes.txt")
CLASSES = ("definition", "example", "step")
DEFAULT_THRESHOLD = 0.23
DEFAULT_MAX_SENTENCES = 400
CACHE_LIMIT = 5000
MIN_WORDS = 3
MAX_WORDS = 80

_SENTENCE_END_RE = re.compile(r"[.!?]+(?:\s+|$)")
# A leading bullet or list number: "- ", "* ", "• ", "1. ", "2) ", "(3) ", "a) ".
_BULLET_MARKS = r"[-*\N{BULLET}\N{BLACK CIRCLE}\N{EN DASH}\N{MIDDLE DOT}]+"
_BULLET_RE = re.compile(r"^\s*(?:" + _BULLET_MARKS + r"|\(?\d{1,3}[.)]|\(?[A-Za-z][.)])\s+")


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> "Array": ...


@dataclass(frozen=True)
class SentenceScore:
    """Cosine similarity of one sentence to each class centroid, rounded to 4 decimals."""

    text: str
    definition: float
    example: float
    step: float


@dataclass(frozen=True)
class ScoreReport:
    """Scores of the sentences that fit the budget, and whether some were left out."""

    scores: list[SentenceScore]
    truncated: bool


def split_sentences(text: str) -> list[str]:
    """Sentences of `text`: split on . ! ? and newlines, bullets stripped, 3-80 words, unique.

    The first occurrence of a sentence (compared case-insensitively) is kept, in text order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        line = _BULLET_RE.sub("", line, count=1)
        start = 0
        pieces: list[str] = []
        for match in _SENTENCE_END_RE.finditer(line):
            pieces.append(line[start : match.end()])
            start = match.end()
        pieces.append(line[start:])
        for piece in pieces:
            sentence = " ".join(piece.split())
            if not MIN_WORDS <= len(sentence.split()) <= MAX_WORDS:
                continue
            key = sentence.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(sentence)
    return out


def load_prototypes() -> dict[str, list[str]]:
    """The prototype sentences per class from the packaged resource file."""
    ref = resources.files("edusync_c2.difficulty").joinpath(*PROTOTYPE_RESOURCE)
    return parse_prototypes(ref.read_text(encoding="utf-8"))


def parse_prototypes(text: str) -> dict[str, list[str]]:
    """Parse ``[section]`` headers, sentence lines and ``#`` comments; every class is required."""
    sections: dict[str, list[str]] = {}
    current: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1].strip().lower(), [])
        elif current is None:
            raise ValueError("prototype sentence before the first [section]")
        else:
            current.append(line)
    missing = [name for name in CLASSES if not sections.get(name)]
    if missing:
        raise ValueError(f"prototype sections missing or empty: {', '.join(missing)}")
    return sections


def _encode(encoder: Encoder, texts: Sequence[str]) -> "Array":
    """Float32 embeddings of `texts`; any encoder failure or wrong shape is a SemanticScoringError.

    The original exception is chained (``__cause__``) but its text, which could echo slide text,
    is never part of the error message.
    """
    import numpy as np

    try:
        vectors = np.asarray(encoder.encode(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("encoder returned an unexpected shape")
    except Exception as exc:
        raise SemanticScoringError() from exc
    return vectors


class _Shared:
    """State shared by a scorer and its per-request views."""

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder
        self.centroids: dict[str, Array] = {}
        self.cache: dict[str, Array] = {}
        self.lock = threading.Lock()


class SemanticScorer:
    """Scores sentences against the definition, example and step prototype centroids."""

    def __init__(
        self,
        encoder: Encoder,
        def_threshold: float = DEFAULT_THRESHOLD,
        ex_threshold: float = DEFAULT_THRESHOLD,
        max_sentences: int = DEFAULT_MAX_SENTENCES,
    ) -> None:
        import numpy as np

        self.def_threshold = def_threshold
        self.ex_threshold = ex_threshold
        self.max_sentences = max_sentences
        self.truncated = False
        self._seen: set[str] = set()
        self._shared = _Shared(encoder)
        prototypes = load_prototypes()
        for name in CLASSES:
            vectors = _encode(encoder, prototypes[name])
            centroid = vectors.mean(axis=0)
            norm = max(float(np.sqrt(np.sum(centroid * centroid))), 1e-12)
            self._shared.centroids[name] = (centroid / norm).astype(np.float32)

    def for_request(self) -> "SemanticScorer":
        """A view with its own sentence budget and ``truncated`` flag; shares centroids, cache."""
        view = object.__new__(SemanticScorer)
        view.def_threshold = self.def_threshold
        view.ex_threshold = self.ex_threshold
        view.max_sentences = self.max_sentences
        view.truncated = False
        view._seen = set()
        view._shared = self._shared
        return view

    def score_report(self, sentences: Sequence[str]) -> ScoreReport:
        """Score `sentences` in order; new distinct sentences beyond the request budget are skipped.

        A sentence already scored in this request is always scored again for free; once
        `max_sentences` distinct sentences are in, further new ones are dropped and `truncated` is
        set (on the report and on this scorer).
        """
        keep: list[str] = []
        truncated = False
        for sentence in sentences:
            if sentence not in self._seen:
                if len(self._seen) >= self.max_sentences:
                    truncated = True
                    continue
                self._seen.add(sentence)
            keep.append(sentence)
        if truncated:
            self.truncated = True
        vectors = self._embed(keep)
        centroids = self._shared.centroids
        scores = [
            SentenceScore(
                text=s,
                definition=round(float(vectors[s] @ centroids["definition"]), 4),
                example=round(float(vectors[s] @ centroids["example"]), 4),
                step=round(float(vectors[s] @ centroids["step"]), 4),
            )
            for s in keep
        ]
        return ScoreReport(scores=scores, truncated=truncated)

    def score_sentences(self, sentences: Sequence[str]) -> list[SentenceScore]:
        """The scores of `sentences` (see ``score_report`` for the budget)."""
        return self.score_report(sentences).scores

    def count_semantic(self, scores: Sequence[SentenceScore]) -> tuple[int, int]:
        """(definition_count, example_count): sentences nearest a class and above its threshold."""
        definitions = examples = 0
        for s in scores:
            if s.definition >= self.def_threshold and s.definition > s.example:
                definitions += 1
            elif s.example >= self.ex_threshold and s.example > s.definition:
                examples += 1
        return definitions, examples

    def _embed(self, sentences: Sequence[str]) -> "dict[str, Array]":
        shared = self._shared
        found: dict[str, Array] = {}
        missing: list[str] = []
        with shared.lock:
            for s in dict.fromkeys(sentences):
                cached = shared.cache.get(s)
                if cached is None:
                    missing.append(s)
                else:
                    found[s] = cached
        if missing:
            encoded = _encode(shared.encoder, missing)
            with shared.lock:
                for s, vector in zip(missing, encoded, strict=True):
                    found[s] = vector
                    if len(shared.cache) >= CACHE_LIMIT:
                        shared.cache.pop(next(iter(shared.cache)))
                    shared.cache[s] = vector
        return found
