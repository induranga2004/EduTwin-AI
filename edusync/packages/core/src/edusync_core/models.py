"""Shared domain models: evidence and flags keyed by concept + lecture timestamp."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LectureTimestamp(BaseModel):
    """A point in a lecture: seconds since lecture start, plus optional wall-clock time."""

    meeting_id: str
    seconds: float = Field(ge=0)
    wall_clock: datetime | None = None


class ConceptRef(BaseModel):
    """A curriculum concept."""

    id: str
    label: str


class EvidenceSource(StrEnum):
    SPEECH = "speech"
    SLIDE = "slide"
    WHITEBOARD = "whiteboard"
    POLL = "poll"
    CHAT = "chat"
    NOTE = "note"


class EvidenceItem(BaseModel):
    """One piece of evidence that a concept was taught, shown or understood."""

    concept: ConceptRef
    source: EvidenceSource
    timestamp: LectureTimestamp
    text: str = ""
    confidence: float = Field(default=1.0, ge=0, le=1)
    meta: dict[str, Any] = Field(default_factory=dict)


class FlagSeverity(StrEnum):
    INFO = "info"
    REVIEW = "review"
    URGENT = "urgent"


class Flag(BaseModel):
    """Something the lecturer should review (gap, difficult concept, low comprehension...)."""

    concept: ConceptRef
    timestamp: LectureTimestamp
    component: str
    reason: str
    severity: FlagSeverity = FlagSeverity.REVIEW
    evidence: list[EvidenceItem] = Field(default_factory=list)
