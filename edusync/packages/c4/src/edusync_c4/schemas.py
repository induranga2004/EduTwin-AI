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
    is_clear: bool = False
    timestamp_valid: bool = True
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


class SlideSnapshot(BaseModel):
    """Reconstructed visual whiteboard state on a slide at a specific point in time."""

    lecture_id: str
    slide_id: str
    timestamp_seconds: float = Field(ge=0.0)
    annotations: list[NormalizedAnnotation] = Field(default_factory=list)
    active_shape_count: int = 0
    slide_text: str | None = None


class AnnotationTimeline(BaseModel):
    """Chronologically sorted timeline of annotation versions with interval bounds."""

    lecture_id: str
    slide_id: str
    events: list[NormalizedAnnotation] = Field(default_factory=list)
    # Events with no usable timestamp: kept for audit, never visible in snapshots.
    unplaced: list[NormalizedAnnotation] = Field(default_factory=list)

    def get_active_annotations(self, timestamp_seconds: float) -> list[NormalizedAnnotation]:
        """Return copies of the annotations active at the given timestamp."""
        active: list[NormalizedAnnotation] = []
        for ann in self.events:
            if ann.is_deleted:
                continue
            if ann.timestamp_seconds > timestamp_seconds:
                continue
            if ann.timestamp_out is not None and timestamp_seconds >= ann.timestamp_out:
                continue
            if ann.timestamp_undo is not None and timestamp_seconds >= ann.timestamp_undo:
                continue
            active.append(ann.model_copy(deep=True))
        return active

    def get_keyframes(self) -> list[float]:
        """Return unique sorted timestamps where annotations appear, change, or disappear."""
        stamps: set[float] = set()
        for ann in self.events:
            stamps.add(ann.timestamp_seconds)
            if ann.timestamp_out is not None:
                stamps.add(ann.timestamp_out)
            if ann.timestamp_undo is not None:
                stamps.add(ann.timestamp_undo)
        return sorted(stamps)
