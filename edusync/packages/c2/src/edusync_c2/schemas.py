"""Request/response models for C2."""

from typing import Self

from pydantic import BaseModel, Field, model_validator

from edusync_core import ConceptRef, Flag, LectureTimestamp


class ComprehensionRequest(BaseModel):
    concept: ConceptRef
    correct: int = Field(ge=0)
    total: int = Field(ge=0)
    timestamp: LectureTimestamp
    predicted_difficulty: float = Field(default=0.0, ge=0, le=1)
    threshold: float = Field(default=0.6, ge=0, le=1)
    min_responses: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _correct_within_total(self) -> Self:
        if self.correct > self.total:
            raise ValueError("correct cannot exceed total")
        return self


class ComprehensionResponse(BaseModel):
    flag: Flag | None = None
