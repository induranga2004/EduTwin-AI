"""C3 HTTP routes."""

from functools import lru_cache

from fastapi import APIRouter, HTTPException

from edusync_c3.keep_filter import KeepFilter, ModelUnavailableError
from edusync_c3.relevance import classify, keyword_overlap_score
from edusync_c3.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    FilterRequest,
    FilterResponse,
    SentenceDecision,
)
from edusync_c3.sentences import split_sentences

router = APIRouter()


@lru_cache
def get_keep_filter() -> KeepFilter:
    """Load the model once; raises ModelUnavailableError (not cached) while it is missing."""
    return KeepFilter.from_dir()


@router.get("/health")
def health() -> dict[str, str]:
    return {"component": "c3", "status": "ok"}


@router.post("/classify")
def classify_text(req: ClassifyRequest) -> ClassifyResponse:
    label = classify(
        req.text,
        req.concepts,
        relevant_threshold=req.relevant_threshold,
        unrelated_threshold=req.unrelated_threshold,
    )
    return ClassifyResponse(label=label, score=keyword_overlap_score(req.text, req.concepts))


@router.post("/filter")
def filter_sentences(req: FilterRequest) -> FilterResponse:
    """Keep only the teaching-content sentences of a transcript."""
    if (req.text is None) == (req.sentences is None):
        raise HTTPException(422, "Provide exactly one of 'text' or 'sentences'")
    sentences = req.sentences if req.sentences is not None else split_sentences(req.text or "")
    try:
        keep_filter = get_keep_filter()
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    threshold = req.threshold if req.threshold is not None else keep_filter.threshold
    scored = [
        SentenceDecision(text=s.text, p_keep=s.p_keep, keep=s.p_keep >= threshold)
        for s in keep_filter.score(sentences)
    ]
    kept = [s.text for s in scored if s.keep]
    return FilterResponse(
        kept=kept,
        total=len(scored),
        kept_count=len(kept),
        threshold=threshold,
        items=scored if req.include_dropped else None,
    )
