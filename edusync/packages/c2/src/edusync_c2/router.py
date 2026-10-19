"""C2 HTTP routes."""

from fastapi import APIRouter

from edusync_c2.comprehension import flag_for_clarification
from edusync_c2.difficulty import lexical_difficulty
from edusync_c2.schemas import (
    ComprehensionRequest,
    ComprehensionResponse,
    DifficultyRequest,
    DifficultyResponse,
)

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"component": "c2", "status": "ok"}


@router.post("/difficulty")
def difficulty(req: DifficultyRequest) -> DifficultyResponse:
    return DifficultyResponse(score=lexical_difficulty(req.text))


@router.post("/comprehension/evaluate")
def evaluate(req: ComprehensionRequest) -> ComprehensionResponse:
    flag = flag_for_clarification(
        req.concept,
        req.correct,
        req.total,
        req.timestamp,
        predicted_difficulty=req.predicted_difficulty,
        threshold=req.threshold,
        min_responses=req.min_responses,
    )
    return ComprehensionResponse(flag=flag)
