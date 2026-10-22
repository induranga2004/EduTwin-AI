"""LOCAL MOCK of C2-style signals (difficulty, comprehension). Not connected to C2.

Both providers are deterministic stand-ins behind Protocols; results are tagged
`source="mock-c2"` so they can never be mistaken for real C2 output.
"""

import re
from typing import Protocol

from edusync_c4.schemas import ComprehensionSignal, DifficultyEstimate

MOCK_SOURCE = "mock-c2"
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


class DifficultyPredictor(Protocol):
    def predict(self, text: str) -> DifficultyEstimate: ...


class ComprehensionEvaluator(Protocol):
    def evaluate(self, concept_id: str) -> ComprehensionSignal | None: ...


class DeterministicDifficultyPredictor:
    """Mock: score grows with average word length (3 letters -> 0, 8+ letters -> 1)."""

    method = "mock-word-length"

    def predict(self, text: str) -> DifficultyEstimate:
        words = _WORD_RE.findall(text)
        if not words:
            return DifficultyEstimate(score=0.0, source=MOCK_SOURCE, method=self.method)
        avg = sum(len(w) for w in words) / len(words)
        score = max(0.0, min(1.0, (avg - 3) / 5))
        return DifficultyEstimate(score=round(score, 3), source=MOCK_SOURCE, method=self.method)


class StaticComprehensionEvaluator:
    """Mock: looks up fixed (correct, total) tallies. Fewer than `min_responses` -> no signal."""

    def __init__(
        self,
        tallies: dict[str, tuple[int, int]],
        threshold: float = 0.6,
        min_responses: int = 3,
    ) -> None:
        self._tallies = tallies
        self._threshold = threshold
        self._min_responses = min_responses

    def evaluate(self, concept_id: str) -> ComprehensionSignal | None:
        tally = self._tallies.get(concept_id)
        if tally is None:
            return None
        correct, total = tally
        if total < self._min_responses or total <= 0 or correct > total:
            return None
        return ComprehensionSignal(
            concept_id=concept_id,
            correct=correct,
            total=total,
            struggling=correct / total < self._threshold,
            source=MOCK_SOURCE,
        )
