"""Request/response and domain models for C4."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

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
    # Full shape after merging all deltas up to this version (style, arrow ends, ...). Contains
    # author metadata, so API responses hide it unless include_raw is requested.
    shape_state: dict[str, Any] = Field(default_factory=dict)


MAX_IMPORT_EVENTS = 5000


class EventImportRequest(BaseModel):
    """Batch ingestion request for annotation events.

    `lecture_id` / `slide_id` are defaults for events that carry no ids of their own.
    `time_unit` applies to the ambiguous `timestamp` field only; `updatedAt` and friends are
    always epoch milliseconds, made lecture-relative by `lecture_start_ms` when given.
    """

    lecture_id: str
    slide_id: str
    events: list[dict[str, Any]] = Field(default_factory=list, max_length=MAX_IMPORT_EVENTS)
    time_unit: Literal["s", "ms"] = "s"
    lecture_start_ms: float | None = None


class EventImportResponse(BaseModel):
    """Batch ingestion response. `imported_count` is the number of newly stored events."""

    imported_count: int
    event_ids: list[str] = Field(default_factory=list)
    deleted_count: int = 0
    clear_count: int = 0
    duplicate_count: int = 0
    conflict_count: int = 0
    conflict_event_ids: list[str] = Field(default_factory=list)
    unplaced_count: int = 0
    unsupported_count: int = 0


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
    # Labels inside a box and arrows attached to shapes always group together, whatever the time.
    structural_links: bool = True


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
    # Boxes-and-arrows read from the group's geometry (plain facts, e.g. 'help -> needed').
    diagram_sentences: list[str] = Field(default_factory=list)

    def evidence_ids(self) -> list[str]:
        ids = [f"annotation:{self.group_id}"]
        ids.extend(f"diagram:{i}" for i in range(len(self.diagram_sentences)))
        if self.slide_evidence_id:
            ids.append(self.slide_evidence_id)
        ids.extend(s.evidence_id for s in self.speech)
        return ids


# --- Underlines, circles and what they point at ---

MAX_SLIDE_WORDS = 5000


class StrokeKind(StrEnum):
    UNDERLINE = "underline"
    LINE = "line"
    CIRCLE = "circle"  # a closed, rounded loop
    LOOP = "loop"  # a closed, boxy loop
    SCRIBBLE = "scribble"
    DOT = "dot"


class SlideWord(BaseModel):
    """One word of the slide text with its box, in whiteboard coordinates."""

    text: str = Field(min_length=1, max_length=200)
    x: float
    y: float
    width: float = Field(ge=0.0)
    height: float = Field(ge=0.0)

    @field_validator("x", "y", "width", "height")
    @classmethod
    def _finite(cls, v: float) -> float:
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("coordinates must be finite numbers")
        return v


class SlideWordsUpsert(BaseModel):
    words: list[SlideWord] = Field(max_length=MAX_SLIDE_WORDS)


class Marking(BaseModel):
    """A recognised freehand mark, and the slide text it points at (if any)."""

    shape_id: str
    kind: StrokeKind
    action: str | None = None  # "underlined" | "circled" | "boxed" when text was found
    target_text: str | None = None
    measurements: dict[str, float] = Field(default_factory=dict)


class RecognizedContent(BaseModel):
    """What can be read from a group without any ML. Placeholders are labelled as such."""

    group_id: str
    kind: str  # "native_text" | "unrecognized_freehand" | "mixed" | "shapes_only" | "empty"
    text: str | None = None
    freehand_stroke_count: int = 0
    is_placeholder: bool = False
    recognizer: str
    notes: list[str] = Field(default_factory=list)
    markings: list[Marking] = Field(default_factory=list)


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


# --- Phase 5: analyze / catch-up API models ---


class AnalyzeRequest(BaseModel):
    """Inputs for the analysis pipeline. Anything omitted falls back to MOCK data."""

    speech: list[SpeechSegment] | None = None
    slide_text: str | None = None
    concepts: list[ConceptRef] | None = None
    speech_pad_seconds: float = Field(default=90.0, ge=0.0)
    speech_offset_seconds: float = 0.0
    use_mock_c2: bool = True


class AnalyzeResponse(BaseModel):
    lecture_id: str
    slide_id: str
    inputs_are_mock: dict[str, bool]
    analyses: list[GroupAnalysis]


class CatchUpItem(BaseModel):
    slide_id: str
    group_id: str
    start_seconds: float
    end_seconds: float
    category: str
    summary: str
    needs_attention: bool
    cited_evidence_ids: list[str]
    limitations: list[str]


class CatchUpResponse(BaseModel):
    lecture_id: str
    since_seconds: float
    items: list[CatchUpItem]
    note: str


# --- Demo Step 1: slides (page info, text and the slide image as SVG) ---

MAX_SLIDE_SVG_CHARS = 5_000_000
MAX_SLIDE_TEXT_CHARS = 200_000


class SlideUpsert(BaseModel):
    """Page info captured from BBB for one slide. All fields optional so partial captures work."""

    number: int | None = Field(default=None, ge=0)
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    scaled_width: float | None = Field(default=None, gt=0)
    scaled_height: float | None = Field(default=None, gt=0)
    text: str | None = Field(default=None, max_length=MAX_SLIDE_TEXT_CHARS)
    svg: str | None = Field(default=None, max_length=MAX_SLIDE_SVG_CHARS)

    @field_validator("svg")
    @classmethod
    def _svg_is_plain_svg(cls, v: str | None) -> str | None:
        if v is None:
            return v
        head = v.lstrip()[:2000].lower()
        if "<svg" not in head:
            raise ValueError("svg must contain an <svg> element")
        lowered = v.lower()
        if "<!doctype" in lowered or "<!entity" in lowered:
            raise ValueError("svg must not contain a DOCTYPE or entity declarations")
        return v


class SlideInfo(BaseModel):
    lecture_id: str
    slide_id: str
    number: int | None = None
    width: float | None = None
    height: float | None = None
    scaled_width: float | None = None
    scaled_height: float | None = None
    text: str | None = None
    has_svg: bool = False


class SlideSeen(BaseModel):
    """One sample of "this slide was on screen at this second" (meeting clock)."""

    seconds: float = Field(ge=0, le=10_000_000)


class SlideVisit(BaseModel):
    """A stretch of time during which one slide was on screen."""

    slide_id: str
    start_seconds: float
    end_seconds: float


# --- Demo Step 3: diagrams (boxes joined by arrows) ---


class DiagramNode(BaseModel):
    shape_id: str
    name: str
    shape_type: ShapeType
    bounds: BoundingBox


class DiagramEdge(BaseModel):
    arrow_shape_id: str
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    relation: str  # "directed" (source -> target) | "both" | "undirected"
    label: str | None = None


class Diagram(BaseModel):
    """Boxes (nodes) and arrows (edges) found on one slide, written out as plain sentences.

    Built from geometry only: arrows attached to shapes, or ending close to a shape, and text
    placed inside a shape. No machine learning is involved.
    """

    diagram_id: str
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    sentences: list[str]
    shape_ids: list[str]
    dangling_arrow_ids: list[str] = Field(default_factory=list)
    first_seconds: float


# --- Demo Step 4: the lecturer's speech ---

MAX_SPEECH_SEGMENTS = 5000
# Whisper often writes the cue word "sync" as one of these.
SYNC_ALIASES = {"sync": ("sync", "sink", "synch", "sinc", "cinq")}


class SpeechUpload(BaseModel):
    """Transcript segments in *recording* time, plus how to line them up with the meeting clock.

    Give `offset_seconds` (meeting time = recording time + offset), or `sync_word`: the lecturer
    said that word at the same moment as drawing the first mark, so the offset is the first
    annotation's time minus the time the word was spoken. With neither, the offset is 0.
    """

    segments: list[SpeechSegment] = Field(max_length=MAX_SPEECH_SEGMENTS)
    offset_seconds: float | None = None
    sync_word: str | None = Field(default=None, min_length=2, max_length=30)
    source: str | None = Field(default=None, max_length=200)


class SpeechInfo(BaseModel):
    lecture_id: str
    segment_count: int
    offset_seconds: float
    method: str  # "manual" | "sync_word" | "none"
    source: str | None = None
    sync_shape_id: str | None = None


class SpeechResponse(SpeechInfo):
    """Stored speech with times already moved onto the meeting clock."""

    segments: list[SpeechSegment]


class LectureSummary(BaseModel):
    lecture_id: str
    event_count: int
    slide_count: int
    first_seconds: float | None = None
    last_seconds: float | None = None
    last_imported: str
    has_speech: bool = False


class ExplainResponse(BaseModel):
    lecture_id: str
    method: str
    slide_count: int
    group_count: int
    used_speech: bool
    used_slide_text: bool
