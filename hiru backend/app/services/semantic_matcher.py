"""
Semantic matching between transcript segments and curriculum concepts.

Uses the pretrained Sentence Transformers MiniLM model. This is inference
with a pretrained model, not training or fine-tuning.

If the model cannot be loaded (for example, no model cache/network access),
the matcher falls back to case-insensitive phrase matching so the API can
continue to work. The response identifies which matching method was used.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SIMILARITY_THRESHOLD = 0.45

_model = None
_model_load_attempted = False


def _load_model():
    """Load MiniLM once, on first semantic-matching request."""
    global _model, _model_load_attempted

    if _model_load_attempted:
        return _model

    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Loaded semantic matching model: %s", MODEL_NAME)
    except Exception:
        logger.exception(
            "Could not load MiniLM; exact phrase matching will be used."
        )
        _model = None

    return _model


def _exact_matches(text: str, concepts: list[dict[str, Any]]) -> list[dict]:
    """Transparent fallback baseline using whole-phrase matching."""
    matches = []
    folded_text = text.casefold()

    for concept in concepts:
        name = str(concept.get("name", "")).strip()
        if not name:
            continue
        pattern = r"(?<!\w)" + re.escape(name.casefold()) + r"(?!\w)"
        if re.search(pattern, folded_text):
            matches.append({
                "name": name,
                "similarity": 1.0,
                "method": "exact_phrase_fallback",
            })

    return matches


def match_concepts_to_text(
    text: str,
    concepts: list[dict[str, Any]],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Match a transcript segment against candidate curriculum concepts.

    Returns match details and the method used. Similarity scores are cosine
    similarity values from the Sentence Transformers model. The default
    threshold is an initial engineering setting and must be calibrated on a
    labelled validation set before being reported as an evaluated result.
    """
    if not text or not text.strip() or not concepts:
        return {
            "matches": [],
            "method": "none",
            "model": None,
            "threshold": threshold,
        }

    model = _load_model()
    if model is None:
        return {
            "matches": _exact_matches(text, concepts),
            "method": "exact_phrase_fallback",
            "model": None,
            "threshold": threshold,
        }

    names = [
        str(concept.get("name", "")).strip()
        for concept in concepts
        if str(concept.get("name", "")).strip()
    ]
    if not names:
        return {
            "matches": [],
            "method": "minilm_cosine_similarity",
            "model": MODEL_NAME,
            "threshold": threshold,
        }

    try:
        embeddings = model.encode(
            [text.strip(), *names],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        text_embedding = embeddings[0]
        similarities = embeddings[1:] @ text_embedding

        matches = []
        for name, score in zip(names, similarities):
            score_value = float(score)
            if score_value >= threshold:
                matches.append({
                    "name": name,
                    "similarity": round(score_value, 4),
                    "method": "minilm_cosine_similarity",
                })

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return {
            "matches": matches,
            "method": "minilm_cosine_similarity",
            "model": MODEL_NAME,
            "threshold": threshold,
        }
    except Exception:
        logger.exception(
            "MiniLM matching failed; exact phrase fallback will be used."
        )
        return {
            "matches": _exact_matches(text, concepts),
            "method": "exact_phrase_fallback",
            "model": None,
            "threshold": threshold,
        }
