"""Semantic matching between transcript segments and curriculum concepts.

Uses the pretrained MiniLM sentence-transformer (inference only, no training). If the model
cannot be loaded, whole-phrase matching is used so the API keeps working; each result
reports which method produced it.
"""

import logging
import re
from collections.abc import Callable
from typing import Any

from edusync_c1.schemas import Concept, ConceptMatch, MatchResult

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.45
SEMANTIC_METHOD = "minilm_cosine_similarity"
EXACT_METHOD = "exact_phrase_fallback"


def _default_loader(model_name: str) -> Any:  # pragma: no cover - needs real weights
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SemanticMatcher:
    """Matches text to concepts by embedding similarity, with an exact-phrase fallback.

    ``loader`` builds the model (injectable for tests); ``enabled=False`` skips the model.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        *,
        enabled: bool = True,
        loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._enabled = enabled
        self._loader = loader or _default_loader
        self._model: Any = None
        self._load_attempted = False

    def _get_model(self) -> Any:
        """Load the model once, on first use; None if unavailable."""
        if not self._enabled:
            return None
        if not self._load_attempted:
            self._load_attempted = True
            try:
                self._model = self._loader(self.model_name)
            except Exception:
                logger.warning("Could not load %s; using exact phrase matching.", self.model_name)
        return self._model

    @staticmethod
    def _names(concepts: list[Concept]) -> list[str]:
        return [c.name.strip() for c in concepts if c.name.strip()]

    def _exact_result(self, text: str, names: list[str], threshold: float) -> MatchResult:
        folded = text.casefold()
        matches = [
            ConceptMatch(name=name, similarity=1.0, method=EXACT_METHOD)
            for name in names
            if re.search(r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)", folded)
        ]
        return MatchResult(matches=matches, method=EXACT_METHOD, threshold=threshold)

    def match(
        self, text: str, concepts: list[Concept], threshold: float | None = None
    ) -> MatchResult:
        """Return concepts mentioned in ``text`` and the method used.

        The default threshold is an engineering setting; calibrate it on labelled data
        before reporting results.
        """
        threshold = self.threshold if threshold is None else threshold
        names = self._names(concepts)
        if not text.strip() or not names:
            return MatchResult(matches=[], method="none", threshold=threshold)

        model = self._get_model()
        if model is None:
            return self._exact_result(text, names, threshold)

        try:
            embeddings = model.encode(
                [text.strip(), *names],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            scores = embeddings[1:] @ embeddings[0]
        except Exception:
            logger.warning("Semantic matching failed; using exact phrase matching.")
            return self._exact_result(text, names, threshold)

        matches = [
            ConceptMatch(name=name, similarity=round(float(score), 4), method=SEMANTIC_METHOD)
            for name, score in zip(names, scores, strict=True)
            if float(score) >= threshold
        ]
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return MatchResult(
            matches=matches, method=SEMANTIC_METHOD, model=self.model_name, threshold=threshold
        )
