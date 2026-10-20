"""Contracts for the pre-lecture difficulty pipeline (Phase 1).

Three groups of models:
- C1 graph input (``C1GraphResponse``): C1's response shape, tolerant of v0 (co-mention only) and
  v1 (prerequisite edges with direction, confidence, source_type). Unknown fields are kept.
- Pipeline records: slides, concepts, features with provenance, fusion input, model output.
- ``DifficultyResponse``: what the API returns; it always says which inputs/stages were real,
  mocked or unavailable, and never contains timings or random ids (same input -> same output).
"""

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

# --------------------------------------------------------------------------- C1 graph (input)


class C1Concept(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str
    frequency: int = 0


class C1Node(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    label: str
    description: str | None = None  # v1
    frequency: int = 0


class C1Evidence(BaseModel):
    model_config = ConfigDict(extra="allow")
    page: int | None = None
    sentence: str = ""


class C1Edge(BaseModel):
    """One relationship. v0: only ``co_mentioned``. v1 adds the prerequisite fields.

    For ``relation_type == "prerequisite"`` the ``source`` concept must be understood before
    ``target`` (``direction == "source_is_prerequisite_of_target"``).
    """

    model_config = ConfigDict(extra="allow")
    source: str
    target: str
    relation_type: str = "co_mentioned"
    direction: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    source_type: str | None = None  # "curriculum_defined" | "inferred"
    evidence_count: int = 0
    evidence: list[C1Evidence] = Field(default_factory=list)


class C1Graph(BaseModel):
    model_config = ConfigDict(extra="allow")
    node_count: int = 0
    edge_count: int = 0
    nodes: list[C1Node] = Field(default_factory=list)
    edges: list[C1Edge] = Field(default_factory=list)
    note: str | None = None


class C1GraphResponse(BaseModel):
    """C1's ``POST /api/c1/lectures/graph`` response, accepted as is."""

    model_config = ConfigDict(extra="allow")
    lecture_id: str | None = None
    filename: str | None = None
    page_count: int | None = None
    concept_count: int | None = None
    concepts: list[C1Concept] = Field(default_factory=list)
    graph: C1Graph
    message: str | None = None


# ---------------------------------------------------------------------- provenance primitives


class FeatureStatus(StrEnum):
    COMPUTED = "computed"  # real value from real inputs
    MOCKED = "mocked"  # placeholder, or computed from a mocked input; never used as real
    UNAVAILABLE = "unavailable"  # cannot be produced (no data, not implemented, not applicable)


class FeatureGroup(StrEnum):
    LANGUAGE = "A_language"
    PREREQUISITES = "B_prerequisites"
    EXPLANATION = "C_explanation"
    DENSITY = "D_density"


class FeatureValue(BaseModel):
    """One feature of one concept, with its provenance."""

    name: str
    group: FeatureGroup
    value: float | None = None
    status: FeatureStatus
    source: str  # e.g. "materials:regex", "c1_graph:prerequisite_edges", "mock:todo_vocab"
    note: str = ""

    @model_validator(mode="after")
    def _value_matches_status(self) -> Self:
        if self.status is FeatureStatus.UNAVAILABLE and self.value is not None:
            raise ValueError("an unavailable feature must not carry a value")
        if self.status is FeatureStatus.COMPUTED and self.value is None:
            raise ValueError("a computed feature needs a value")
        return self


class InputSource(BaseModel):
    """Where an input came from, so mocks are always visible in the response."""

    name: str  # "materials" | "c1_graph"
    kind: Literal["real", "mock", "none"]
    detail: str = ""


class StageStatus(BaseModel):
    name: str
    status: Literal["ok", "partial", "mocked", "skipped", "failed"]
    detail: str = ""


# ------------------------------------------------------------------------------- materials


class ParagraphRecord(BaseModel):
    text: str
    level: int | None = None  # bullet/indent level (0 = top); None when unknown (PDF)


class SlideRecord(BaseModel):
    index: int = Field(ge=1)  # 1-based slide or page number
    title: str | None = None
    text: str = ""  # all text of the slide (title included), notes excluded
    notes: str = ""
    paragraphs: list[ParagraphRecord] = Field(default_factory=list)
    picture_count: int | None = None  # None = unknown (PDF)
    chart_count: int | None = None
    table_count: int | None = None
    shape_count: int | None = None
    formula_hint: bool = False  # text looks like it contains a formula or equation
    code_hint: bool = False  # text looks like source code


class MaterialsDoc(BaseModel):
    filename: str
    file_type: Literal["pptx", "pdf"]
    slide_count: int
    structure_available: bool  # False for PDF: only text per page is known
    slides: list[SlideRecord]
    warnings: list[str] = Field(default_factory=list)


# -------------------------------------------------------------------------------- concepts


class AlignmentStatus(StrEnum):
    ALIGNED = "aligned"  # concept found on at least one slide
    NOT_FOUND = "not_found_in_materials"


class ConceptRecord(BaseModel):
    id: str  # C1 node id, used as the canonical concept id
    label: str
    description: str | None = None
    frequency: int = 0  # C1 node frequency
    slide_indexes: list[int] = Field(default_factory=list)
    alignment_status: AlignmentStatus
    filter_status: Literal["kept", "dropped"] = "kept"
    filter_note: str = ""


class GraphContext(BaseModel):
    """The graph the pipeline works with plus where it came from."""

    response: C1GraphResponse
    source: InputSource


# ------------------------------------------------------------------------------- fusion/model


class DifficultyLabel(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class FusionFeature(BaseModel):
    name: str
    value: float | None = None  # None when missing
    status: FeatureStatus
    missing: bool  # True for unavailable features and (by default) mocked ones


class FusionInput(BaseModel):
    """Model input: text plus the ordered structured features (registry order)."""

    concept_id: str
    text: str
    features: list[FusionFeature]


class ModelKind(StrEnum):
    BASELINE = "baseline"  # rule-based, not trained
    TRAINED = "trained"
    TODO = "todo"  # placeholder class, not implemented


class ModelInfo(BaseModel):
    name: str
    version: str
    kind: ModelKind
    description: str
    trained_on: str | None = None


class ModelOutput(BaseModel):
    label: DifficultyLabel
    score: float | None = Field(default=None, ge=0, le=1)  # raw baseline score, not a probability
    probabilities: dict[DifficultyLabel, float] | None = None  # None until a trained model exists
    explanation: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------------- response


class GroupSummary(BaseModel):
    computed: int = 0
    mocked: int = 0
    unavailable: int = 0


class ConceptPrediction(BaseModel):
    concept_id: str
    label: str
    difficulty: DifficultyLabel | None
    score: float | None = None
    probabilities: dict[DifficultyLabel, float] | None = None
    confidence: float | None = None  # always None in Phase 1: no calibrated confidence exists
    alignment_status: AlignmentStatus
    slide_indexes: list[int] = Field(default_factory=list)
    features: list[FeatureValue]
    feature_summary: dict[FeatureGroup, GroupSummary]
    status: Literal["ok", "partial", "failed"]  # partial = some features mocked/unavailable
    explanation: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DifficultyResponse(BaseModel):
    pipeline_version: str = "0.1.0-phase1"
    lecture_id: str | None = None
    filename: str
    file_type: Literal["pptx", "pdf"]
    slide_count: int
    inputs: list[InputSource]
    stages: list[StageStatus]
    model: ModelInfo
    predictions: list[ConceptPrediction]
    todo: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Phase 1 baseline: rule-based labels from simple features, not a trained model. "
        "No probabilities or accuracy are claimed; mocked inputs and features are labelled."
    )


class FeatureSpec(BaseModel):
    name: str
    group: FeatureGroup
    phase1: Literal["real", "real_needs_structure", "graph", "mock", "todo"]
    description: str


class DifficultyInfo(BaseModel):
    model: ModelInfo
    graph_provider: str
    concept_filter: str
    features: list[FeatureSpec]
    todo: list[str]
