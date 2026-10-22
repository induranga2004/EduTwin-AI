"""Request/response and domain models for C4."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from edusync_core import ConceptRef, EvidenceItem, Flag


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


class GroupingConfig(BaseModel):
    """Thresholds for rule-based candidate grouping (virtual slide units / seconds)."""

    max_distance: float = Field(default=50.0, ge=0.0)
    max_gap_seconds: float = Field(default=15.0, ge=0.0)
    include_retracted: bool = True


class AnnotationGroup(BaseModel):
    """A *candidate* set of annotations that are close in space and time on one slide.

    This is raw structural evidence only. It carries no claim about what the group means
    (equation, diagram, underline...); any interpretation belongs to a later, separate layer.
    """

    group_id: str
    lecture_id: str
    slide_id: str
    method: str
    shape_ids: list[str]
    event_ids: list[str]
    retracted_shape_ids: list[str] = Field(default_factory=list)
    member_count: int
    primary_type: ShapeType
    bounds: BoundingBox
    start_seconds: float
    end_seconds: float
    native_text: str | None = None


# --- Phase 4: context alignment, recognition placeholder, interpretation ---
# Evidence models (SpeechEvidence, AnnotationContext) only record what was observed.
# Derived/rule-based output lives in separate models (RecognizedContent, AnnotationInterpretation).


class SpeechSegment(BaseModel):
    """A timestamped transcript segment, in seconds from the start of the recording."""

    id: int | str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str


class SlideText(BaseModel):
    """Whole-slide text. No coordinates are available, so it cannot be matched spatially."""

    slide_id: str
    text: str


class ConceptHit(BaseModel):
    """A concept name found inside some text by a ConceptMatcher."""

    concept: ConceptRef
    matched_text: str


class ConceptCandidate(BaseModel):
    """A candidate link between evidence and a concept. Mention only, not coverage or mastery."""

    concept: ConceptRef
    matched_text: str
    source: str  # "annotation" | "slide" | "speech"
    evidence_id: str
    method: str


class SpeechEvidence(BaseModel):
    evidence_id: str
    segment_id: int | str
    start: float
    end: float
    text: str
    relation: str  # "during" | "before" | "after" the group's drawing span


class AnnotationContext(BaseModel):
    """Observed evidence around one annotation group. Contains no interpretation."""

    group_id: str
    lecture_id: str
    slide_id: str
    span_start: float
    span_end: float
    window_start: float
    window_end: float
    native_text: str | None = None
    slide_evidence_id: str | None = None
    slide_text_excerpt: str | None = None
    slide_text_truncated: bool = False
    speech: list[SpeechEvidence] = Field(default_factory=list)
    concept_candidates: list[ConceptCandidate] = Field(default_factory=list)

    def evidence_ids(self) -> list[str]:
        ids = [f"annotation:{self.group_id}"]
        if self.slide_evidence_id:
            ids.append(self.slide_evidence_id)
        ids.extend(s.evidence_id for s in self.speech)
        return ids


class RecognizedContent(BaseModel):
    """What can be read from a group without any ML. Placeholders are labelled as such."""

    group_id: str
    kind: str  # "native_text" | "unrecognized_freehand" | "mixed" | "shapes_only" | "empty"
    text: str | None = None
    freehand_stroke_count: int = 0
    is_placeholder: bool = False
    recognizer: str
    notes: list[str] = Field(default_factory=list)


class DifficultyEstimate(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    source: str  # always names the provider, e.g. "mock-c2"
    method: str


class ComprehensionSignal(BaseModel):
    concept_id: str
    correct: int = Field(ge=0)
    total: int = Field(ge=0)
    struggling: bool
    source: str


class AnnotationInterpretation(BaseModel):
    """Rule-based, template-built explanation. Derived from evidence; not AI understanding."""

    group_id: str
    category: str
    summary: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    method: str
    confidence: float = Field(ge=0.0, le=1.0)
    difficulty: DifficultyEstimate | None = None
    comprehension: list[ComprehensionSignal] = Field(default_factory=list)
    needs_attention: bool = False
    limitations: list[str] = Field(default_factory=list)


class GroupAnalysis(BaseModel):
    """Evidence and derived output for one group, kept side by side but as separate objects."""

    context: AnnotationContext
    recognized: RecognizedContent
    interpretation: AnnotationInterpretation
