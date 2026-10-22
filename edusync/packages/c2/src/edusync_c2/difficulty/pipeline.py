"""The pre-lecture difficulty pipeline: materials -> graph -> concepts -> features -> model.

Deterministic: no timings, ids or clocks in the response.
"""

import time
from collections.abc import Callable

from edusync_c2.difficulty.concepts import (
    ConceptFilter,
    ContextualConceptFilter,
    MockConceptFilter,
    build_concepts,
)
from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.features.semantic import SemanticScorer
from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer
from edusync_c2.difficulty.fusion import build_fusion_input, summarize_features
from edusync_c2.difficulty.graph import GraphView
from edusync_c2.difficulty.materials import MaterialsError, parse_materials
from edusync_c2.difficulty.models import DifficultyModel, RuleBaselineModel
from edusync_c2.difficulty.providers import (
    GraphProvider,
    MockC1GraphProvider,
    PayloadGraphProvider,
)
from edusync_c2.difficulty.registry import (
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    PHASE1_TODO,
    feature_schema_hash,
)
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    C1GraphResponse,
    ConceptPrediction,
    DifficultyInfo,
    DifficultyResponse,
    FeatureStatus,
    InputSource,
    ModelKind,
    StageStatus,
)
from edusync_c2.logger import get_logger

logger = get_logger(__name__)

CONCEPT_CYCLE_WARNING = "prerequisite cycle found below this concept; the back edge was ignored"
NO_PREDICTION_WARNING = "not enough computed features for the baseline"
NO_PREREQ_WARNING = "C1 graph has no prerequisite edges: group B features unavailable"
SEMANTIC_UNAVAILABLE = "minilm:unavailable"
SEMANTIC_UNAVAILABLE_WARNING = "semantic scorer unavailable: semantic features are unavailable"
SEMANTIC_TRUNCATED_WARNING = (
    "semantic scorer sentence budget reached: later sentences were not scored, "
    "so semantic counts are lower bounds"
)
MOCK_GRAPH_WARNING = (
    "C1 graph is a mock, not C1 output for this lecture (see inputs): "
    "group B and frequency features are mocked"
)
CURATED_GRAPH_WARNING = (
    "C1 graph is hand-made (curated JSON), not C1 output: "
    "prerequisite values are only as good as the curation"
)


class DifficultyPipeline:
    def __init__(
        self,
        *,
        model: DifficultyModel | None = None,
        concept_filter: ConceptFilter | ContextualConceptFilter | None = None,
        mock_provider: GraphProvider | None = None,
        max_mb: int = 100,
        use_mocked_features: bool = False,
        selected_model: str = "baseline",
        vocab: VocabularyAnalyzer | None = None,
        graph_provider_factory: Callable[[bytes, str], GraphProvider] | None = None,
        graph_provider_name: str | None = None,
        semantic: SemanticScorer | None = None,
        semantic_status: str = "off",
    ) -> None:
        self.semantic = semantic
        self.semantic_status = semantic_status
        self.graph_provider_factory = graph_provider_factory
        self.graph_provider_name = graph_provider_name
        self.selected_model = selected_model
        self.vocab = vocab
        self.model: DifficultyModel = model or RuleBaselineModel()
        self.concept_filter: ConceptFilter | ContextualConceptFilter = (
            concept_filter or MockConceptFilter()
        )
        self.mock_provider: GraphProvider = mock_provider or MockC1GraphProvider()
        self.max_mb = max_mb
        self.use_mocked_features = use_mocked_features

    def info(self) -> DifficultyInfo:
        return DifficultyInfo(
            model=self.model.info,
            graph_provider=self.graph_provider_name or self.mock_provider.name,
            concept_filter=self.concept_filter.name,
            features=list(FEATURE_SPECS),
            todo=list(PHASE1_TODO),
            selected_model=self.selected_model,
            feature_schema_hash=feature_schema_hash(),
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            semantic_scorer=self.semantic_status,
        )

    def run(
        self, data: bytes, filename: str, c1_graph: C1GraphResponse | None = None
    ) -> DifficultyResponse:
        started = time.perf_counter()
        try:
            materials = parse_materials(data, filename, max_mb=self.max_mb)
        except MaterialsError as exc:
            logger.warning("materials_rejected status=%d reason=%s", exc.status_code, exc.message)
            raise
        logger.info(
            "materials type=%s bytes=%d slides=%d warnings=%d",
            materials.file_type,
            len(data),
            materials.slide_count,
            len(materials.warnings),
        )
        # A posted graph always wins; then the configured factory; else the slide-title mock.
        provider: GraphProvider
        if c1_graph is not None:
            provider = PayloadGraphProvider(c1_graph)
        elif self.graph_provider_factory is not None:
            provider = self.graph_provider_factory(data, filename)
        else:
            provider = self.mock_provider
        ctx = provider.build(materials)
        raw_view = GraphView(ctx.response)
        graph = ctx.response.graph
        node_ids = {n.id for n in graph.nodes}
        prereq_total = sum(1 for e in graph.edges if e.relation_type == "prerequisite")
        prereq_valid = sum(len(raw_view.prerequisites_of(i)) for i in node_ids)
        logger.info(
            "graph source=%s nodes=%d edges=%d prerequisite_edges=%d ignored=%d",
            ctx.source.kind,
            len(graph.nodes),
            len(graph.edges),
            prereq_valid,
            prereq_total - prereq_valid,
        )
        if raw_view.warnings:  # the texts name concept ids, so they are DEBUG only
            logger.warning("graph_warning count=%d", len(raw_view.warnings))
            for w in raw_view.warnings:
                logger.debug("graph_warning_detail %s", w)
        all_found = build_concepts(ctx, materials, self.concept_filter)
        concepts = [c for c in all_found if c.filter_status == "kept"]
        n_aligned = sum(1 for c in concepts if c.alignment_status is AlignmentStatus.ALIGNED)
        aliases = {a: c.id for c in all_found for a in c.merged_from}
        dropped = {c.id for c in all_found if c.filter_status == "dropped"}
        view = (
            GraphView(ctx.response, aliases=aliases, dropped=dropped)
            if aliases or dropped
            else raw_view
        )
        merged_note = f" merged={len(aliases)}" if aliases else ""
        logger.info(
            "concepts total=%d aligned=%d not_found=%d filter=%s dropped=%d%s",
            len(concepts),
            n_aligned,
            len(concepts) - n_aligned,
            self.concept_filter.name,
            len(all_found) - len(concepts),
            merged_note,
        )
        logger.debug("concepts_detail ids=%s", [c.id for c in concepts])
        slides_by_index = {s.index: s for s in materials.slides}
        # A per-request view: shared embeddings, own sentence budget (see SemanticScorer).
        request_scorer = self.semantic.for_request() if self.semantic is not None else None
        slide_size = (
            (materials.slide_width_emu, materials.slide_height_emu)
            if materials.slide_width_emu is not None and materials.slide_height_emu is not None
            else None
        )

        all_features = []
        fusion_inputs = []
        cyclic_ids: set[str] = set()
        for c in concepts:
            metrics = view.metrics(c.id)
            if metrics is not None and metrics.has_cycle:
                cyclic_ids.add(c.id)
            features = compute_all_features(
                c,
                all_concepts=concepts,
                slides_by_index=slides_by_index,
                structure_available=materials.structure_available,
                metrics=metrics,
                graph_has_prerequisite_edges=view.has_prerequisite_edges,
                graph_source=ctx.source,
                vocab=self.vocab,
                slide_size=slide_size,
                semantic=request_scorer,
                semantic_status=self.semantic_status,
            )
            all_features.append(features)
            fusion_inputs.append(
                build_fusion_input(
                    c,
                    features,
                    [
                        slides_by_index[i]
                        for i in sorted(set(c.slide_indexes))
                        if i in slides_by_index
                    ],
                    use_mocked=self.use_mocked_features,
                )
            )
        flat = [f for fs in all_features for f in fs]
        logger.info(
            "features concepts=%d computed=%d mocked=%d unavailable=%d",
            len(concepts),
            sum(1 for f in flat if f.status is FeatureStatus.COMPUTED),
            sum(1 for f in flat if f.status is FeatureStatus.MOCKED),
            sum(1 for f in flat if f.status is FeatureStatus.UNAVAILABLE),
        )
        logger.info(
            "fusion inputs=%d features_per_input=%d", len(fusion_inputs), len(FEATURE_SPECS)
        )
        if cyclic_ids:
            logger.warning("graph_cycle concepts=%d", len(cyclic_ids))
        outputs = self.model.predict(fusion_inputs)
        logger.info(
            "model name=%s kind=%s predicted=%d none=%d",
            self.model.info.name,
            self.model.info.kind.value,
            sum(1 for o in outputs if o is not None),
            sum(1 for o in outputs if o is None),
        )

        predictions: list[ConceptPrediction] = []
        for c, features, out in zip(concepts, all_features, outputs, strict=True):
            warnings: list[str] = []
            if c.id in cyclic_ids:
                warnings.append(CONCEPT_CYCLE_WARNING)
            if c.alignment_status is AlignmentStatus.NOT_FOUND:
                warnings.append("concept not found in the lecture materials")
            if out is None:
                warnings.append(NO_PREDICTION_WARNING)
                status = "failed"
            elif all(f.status is FeatureStatus.COMPUTED for f in features):
                status = "ok"
            else:
                status = "partial"
            predictions.append(
                ConceptPrediction(
                    concept_id=c.id,
                    label=c.label,
                    difficulty=out.label if out else None,
                    score=out.score if out else None,
                    probabilities=out.probabilities if out else None,
                    confidence=None,
                    alignment_status=c.alignment_status,
                    slide_indexes=list(c.slide_indexes),
                    merged_from=list(c.merged_from),
                    features=features,
                    feature_summary=summarize_features(features),
                    status=status,
                    explanation=list(out.explanation) if out else [],
                    warnings=warnings,
                )
            )

        mock_graph = ctx.source.kind == "mock"
        warnings = list(materials.warnings) + list(raw_view.warnings)
        warnings += [w for w in view.warnings if w not in raw_view.warnings]
        if mock_graph:
            warnings.append(MOCK_GRAPH_WARNING)
        if ctx.source.kind == "curated":
            warnings.append(CURATED_GRAPH_WARNING)
        if not view.has_prerequisite_edges:
            warnings.append(NO_PREREQ_WARNING)
        if cyclic_ids:
            warnings.append(
                f"{len(cyclic_ids)} concept(s) have a prerequisite cycle in the C1 graph"
            )
        if self.semantic_status == SEMANTIC_UNAVAILABLE:
            warnings.append(SEMANTIC_UNAVAILABLE_WARNING)
        if request_scorer is not None and request_scorer.truncated:
            logger.warning("semantic_truncated limit=%d", request_scorer.max_sentences)
            warnings.append(SEMANTIC_TRUNCATED_WARNING)
        logger.info(
            "done concepts=%d duration_ms=%d",
            len(predictions),
            round((time.perf_counter() - started) * 1000),
        )

        # Caller payload: echo its id. Configured file/c1_api provider: the provider's own
        # (deterministic) id. Default mock: none, as before.
        if c1_graph is not None:
            lecture_id = c1_graph.lecture_id
        elif self.graph_provider_factory is not None:
            lecture_id = ctx.response.lecture_id
        else:
            lecture_id = None
        return DifficultyResponse(
            lecture_id=lecture_id,
            filename=materials.filename,
            file_type=materials.file_type,
            slide_count=materials.slide_count,
            inputs=[
                InputSource(
                    name="materials",
                    kind="real",
                    detail=(
                        f"{materials.filename} ({materials.file_type}, "
                        f"{materials.slide_count} slides)"
                    ),
                ),
                ctx.source,
            ],
            stages=self._stages(
                materials,
                ctx.source,
                view,
                concepts,
                all_features,
                predictions,
                len(cyclic_ids),
                merged=(view is not raw_view, len(aliases), len(dropped)),
            ),
            model=self.model.info,
            predictions=predictions,
            todo=list(PHASE1_TODO),
            warnings=warnings,
        )

    def _stages(
        self,
        materials,
        source: InputSource,
        view,
        concepts,
        all_features,
        predictions,
        n_cycles=0,
        merged=(False, 0, 0),
    ):
        mock_graph = source.kind == "mock"
        graph_detail = source.detail
        if view.warnings:
            graph_detail += "; " + "; ".join(view.warnings)
        if n_cycles:
            graph_detail += f"; {n_cycles} concept(s) with a prerequisite cycle"
        if merged[0]:
            parts = [
                f"{n} {word}"
                for n, word in (
                    (view.merged_aliases, "aliases merged"),
                    (view.rewritten_edges, "edges rewritten"),
                    (view.dropped_edges, "edges dropped"),
                )
                if n
            ]
            if parts:
                graph_detail += "; " + ", ".join(parts)
        aligned = sum(1 for c in concepts if c.alignment_status is AlignmentStatus.ALIGNED)
        concept_detail = f"{len(concepts)} concepts, {aligned} aligned, "
        concept_detail += f"{len(concepts) - aligned} not found in materials"
        mock_filter = bool(getattr(self.concept_filter, "is_mock", False))
        if not mock_filter and (merged[1] or merged[2]):
            concept_detail += f"; filter {self.concept_filter.name}: "
            concept_detail += f"{merged[1]} merged, {merged[2]} dropped"
        degraded = any(f.status is not FeatureStatus.COMPUTED for fs in all_features for f in fs)
        n_failed = sum(1 for p in predictions if p.status == "failed")
        kind = self.model.info.kind
        model_detail = f"{self.model.info.name} ({kind.value})"
        if kind is ModelKind.BASELINE:
            model_detail += ": rule-based baseline, not trained"
        if n_failed:
            model_detail += f"; {n_failed} concepts without a prediction"
        return [
            StageStatus(
                name="materials",
                status="partial" if materials.warnings else "ok",
                detail=f"{materials.slide_count} slides parsed"
                + (f"; {'; '.join(materials.warnings)}" if materials.warnings else ""),
            ),
            StageStatus(name="graph", status="mocked" if mock_graph else "ok", detail=graph_detail),
            StageStatus(
                name="concepts",
                status="mocked" if mock_filter else "ok",
                detail=concept_detail
                + ("; concept filter is a mock (keeps all)" if mock_filter else ""),
            ),
            StageStatus(
                name="features",
                status="partial" if degraded else "ok",
                detail=("some features are mocked or unavailable" if degraded else "all computed"),
            ),
            StageStatus(name="fusion", status="ok", detail="ordered features with missing masks"),
            StageStatus(name="model", status="ok", detail=model_detail),
        ]
