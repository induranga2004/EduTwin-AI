"""Typed request/response models for the C1 API (mirror the prototype JSON shapes)."""

from pydantic import BaseModel, Field


class Page(BaseModel):
    """Text of one PDF page or PPTX slide."""

    page: int
    text: str


class Concept(BaseModel):
    """A candidate concept and how often it occurs in the document."""

    name: str
    frequency: int = 0


class ConceptNode(BaseModel):
    id: str
    label: str
    frequency: int = 0
    description: str = ""
    depth: int = 1
    is_main_concept: bool = False


class EdgeEvidence(BaseModel):
    page: int | None = None
    sentence: str


class ConceptEdge(BaseModel):
    """Curriculum prerequisite or co-mention edge."""

    source: str
    target: str
    relation_type: str = "co_mentioned"
    direction: str = "source_is_prerequisite_of_target"
    confidence: float = 1.0
    source_type: str = "curriculum_defined"
    evidence_count: int = 0
    evidence: list[EdgeEvidence] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    node_count: int
    edge_count: int
    nodes: list[ConceptNode]
    edges: list[ConceptEdge]
    note: str
    main_concept: str | None = None


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """Normalised speech-to-text result."""

    segments: list[TranscriptSegment]
    language: str | None = None
    text: str = ""


class ConceptMatch(BaseModel):
    name: str
    similarity: float
    method: str


class MatchResult(BaseModel):
    matches: list[ConceptMatch]
    method: str
    model: str | None = None
    threshold: float


class TimelineNode(ConceptNode):
    """Graph node annotated with speech evidence seen so far."""

    evidence_count: int = 0
    status: str = "not_yet_observed"
    latest_similarity: float | None = None


class TimelineGraph(BaseModel):
    node_count: int
    edge_count: int
    nodes: list[TimelineNode]
    edges: list[ConceptEdge]
    note: str


class TimelineStep(BaseModel):
    """Cumulative graph snapshot after one transcript segment."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    transcript_text: str
    matched_concepts: list[str]
    concept_matches: list[ConceptMatch]
    matching_method: str
    nodes: list[TimelineNode]
    edges: list[ConceptEdge]


class MatchingInfo(BaseModel):
    preferred_method: str
    model: str
    similarity_threshold: float
    fallback: str


class GraphTimeline(BaseModel):
    initial_graph: TimelineGraph
    timeline: list[TimelineStep]
    matching: MatchingInfo
    note: str


class HealthResponse(BaseModel):
    status: str = "ok"
    component: str = "c1"


class UploadResponse(BaseModel):
    lecture_id: str
    filename: str | None
    page_count: int
    extracted_character_count: int
    concept_count: int
    concepts: list[Concept]
    pages: list[Page]
    message: str


class GraphResponse(BaseModel):
    lecture_id: str
    filename: str | None
    page_count: int
    concept_count: int
    concepts: list[Concept]
    graph: KnowledgeGraph
    message: str


class TranscribeResponse(BaseModel):
    recording_id: str
    filename: str | None
    document_filename: str | None
    language: str
    segment_count: int
    transcript: str
    segments: list[TranscriptSegment]
    concept_count: int
    concepts: list[Concept]
    graph: KnowledgeGraph
    graph_timeline: GraphTimeline
    message: str
