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


class FilterRequest(BaseModel):
    """Either ``text`` (a transcript, split into sentences here) or ``sentences``."""

    text: str | None = None
    sentences: list[str] | None = None
    threshold: float | None = Field(default=None, gt=0, lt=1)
    include_dropped: bool = False


class SentenceDecision(BaseModel):
    text: str
    p_keep: float = Field(ge=0, le=1)
    keep: bool


class FilterResponse(BaseModel):
    kept: list[str]
    total: int
    kept_count: int
    threshold: float
    items: list[SentenceDecision] | None = None
