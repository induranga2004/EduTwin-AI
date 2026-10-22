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


class LiveEvalConceptInput(BaseModel):
    id: str
    title: str
    slideNum: int = 1
    index: int = 0
    points: list[str] = Field(default_factory=list)
    description: str = ""


class LiveEvalRequest(BaseModel):
    utterance_text: str
    timestamp_seconds: float = 0.0
    delivered_concept_ids: list[str] = Field(default_factory=list)
    concepts: list[LiveEvalConceptInput] = Field(default_factory=list)
    main_concept_id: str | None = None


class LiveEvalResponse(BaseModel):
    matched_concept_id: str | None = None
    matched_concept_title: str | None = None
    matched_slide_num: int | None = None
    semantic_similarity: float = 0.0
    matching_method: str = "minilm_semantic_model"
    model_status: str = "not_yet_observed"
    model_confidence: float = 1.0
    class_probabilities: dict[str, float] = Field(default_factory=dict)
    prereq_satisfied: bool = True
    skipped_prerequisites: list[str] = Field(default_factory=list)
    main_concept_synthesized: bool = False
    newly_delivered_ids: list[str] = Field(default_factory=list)
    explanation: str = ""

