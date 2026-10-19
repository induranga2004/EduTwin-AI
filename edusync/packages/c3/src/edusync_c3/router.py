"""C3 HTTP routes."""

from fastapi import APIRouter

from edusync_c3.relevance import classify, keyword_overlap_score
from edusync_c3.schemas import ClassifyRequest, ClassifyResponse

router = APIRouter()


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
