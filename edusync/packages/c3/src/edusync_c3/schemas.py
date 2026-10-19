"""Request/response models for C3."""

from pydantic import BaseModel, Field

from edusync_c3.relevance import RelevanceLabel


class ClassifyRequest(BaseModel):
    text: str
    concepts: list[str]
    relevant_threshold: float = Field(default=0.2, ge=0, le=1)
    unrelated_threshold: float = Field(default=0.05, ge=0, le=1)


class ClassifyResponse(BaseModel):
    label: RelevanceLabel
    score: float = Field(ge=0, le=1)
    method: str = "keyword-overlap"
