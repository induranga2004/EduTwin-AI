"""Request/response and domain models for C4."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from edusync_core import EvidenceItem, Flag


class ReconcileRequest(BaseModel):
    speech: list[EvidenceItem] = Field(default_factory=list)
    visual: list[EvidenceItem] = Field(default_factory=list)
    window_seconds: float = Field(default=120.0, gt=0)


class ReconcileResponse(BaseModel):
    flags: list[Flag]


class ShapeType(StrEnum):
    """Supported whiteboard annotation shape types."""

    DRAW = "draw"
    TEXT = "text"
    GEO = "geo"
    ARROW = "arrow"
    LINE = "line"
    NOTE = "note"
    HIGHLIGHT = "highlight"
    FRAME = "frame"
    POLL = "poll"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box in virtual slide coordinates (pixels)."""

    x: float
    y: float
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)


class StrokePoint(BaseModel):
    """A point in a freehand stroke or polyline."""

    x: float
    y: float
    pressure: float = 0.5


class NormalizedAnnotation(BaseModel):
    """Normalized annotation event with exact geometry, content, and temporal bounds."""

    event_id: str
    lecture_id: str
    slide_id: str
    shape_id: str
    shape_type: ShapeType
    timestamp_seconds: float = Field(ge=0.0)
    timestamp_out: float | None = None
    timestamp_undo: float | None = None
    bounds: BoundingBox
    stroke_points: list[StrokePoint] = Field(default_factory=list)
    text_content: str | None = None
    is_deleted: bool = False
    user_id: str = "unknown"
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class EventImportRequest(BaseModel):
    """Batch ingestion request for annotation events."""

    lecture_id: str
    slide_id: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class EventImportResponse(BaseModel):
    """Batch ingestion response."""

    imported_count: int
    event_ids: list[str] = Field(default_factory=list)
    deleted_count: int = 0
