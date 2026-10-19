"""Request/response models for C4."""

from pydantic import BaseModel, Field

from edusync_core import EvidenceItem, Flag


class ReconcileRequest(BaseModel):
    speech: list[EvidenceItem] = Field(default_factory=list)
    visual: list[EvidenceItem] = Field(default_factory=list)
    window_seconds: float = Field(default=120.0, gt=0)


class ReconcileResponse(BaseModel):
    flags: list[Flag]
