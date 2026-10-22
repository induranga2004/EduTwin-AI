"""C3 HTTP routes."""

from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query, Request

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


def _filter(sentences: list[str], threshold: float | None, include_dropped: bool) -> FilterResponse:
    try:
        keep_filter = get_keep_filter()
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc
    cutoff = threshold if threshold is not None else keep_filter.threshold
    scored = [
        SentenceDecision(text=s.text, p_keep=s.p_keep, keep=s.p_keep >= cutoff)
        for s in keep_filter.score(sentences)
    ]
    kept = [s.text for s in scored if s.keep]
    return FilterResponse(
        kept=kept,
        total=len(scored),
        kept_count=len(kept),
        threshold=cutoff,
        items=scored if include_dropped else None,
    )


@router.post("/filter")
def filter_sentences(req: FilterRequest) -> FilterResponse:
    """Keep only the teaching-content sentences of a transcript."""
    if (req.text is None) == (req.sentences is None):
        raise HTTPException(422, "Provide exactly one of 'text' or 'sentences'")
    sentences = req.sentences if req.sentences is not None else split_sentences(req.text or "")
    return _filter(sentences, req.threshold, req.include_dropped)


@router.post(
    "/filter-text",
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"text/plain": {"schema": {"type": "string"}}},
        }
    },
)
async def filter_text(
    request: Request,
    threshold: float | None = Query(default=None, gt=0, lt=1),
    include_dropped: bool = False,
) -> FilterResponse:
    """Same as ``/filter``, but the body is the raw transcript (no JSON escaping needed)."""
    try:
        text = (await request.body()).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(422, "The body must be UTF-8 text") from exc
    if not text.strip():
        raise HTTPException(422, "The body is empty: send the transcript as plain text")
    return _filter(split_sentences(text), threshold, include_dropped)
