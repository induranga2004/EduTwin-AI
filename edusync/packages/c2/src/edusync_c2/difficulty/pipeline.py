"""The pre-lecture difficulty pipeline: materials -> graph -> concepts -> features -> model.

Deterministic: no timings, ids or clocks in the response.
"""

from edusync_c2.difficulty.concepts import ConceptFilter, MockConceptFilter, build_concepts
from edusync_c2.difficulty.features import compute_all_features
from edusync_c2.difficulty.fusion import build_fusion_input, summarize_features
from edusync_c2.difficulty.graph import GraphView
from edusync_c2.difficulty.materials import parse_materials
from edusync_c2.difficulty.models import DifficultyModel, RuleBaselineModel
from edusync_c2.difficulty.providers import (
    GraphProvider,
    MockC1GraphProvider,
    PayloadGraphProvider,
)
from edusync_c2.difficulty.registry import FEATURE_SPECS, PHASE1_TODO
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

NO_PREDICTION_WARNING = "not enough computed features for the baseline"
NO_PREREQ_WARNING = "C1 graph has no prerequisite edges: group B features unavailable"
MOCK_GRAPH_WARNING = (
    "C1 graph is a mock built from slide titles: group B and frequency features are mocked"
)


class DifficultyPipeline:
    def __init__(
        self,
        *,
        model: DifficultyModel | None = None,
        concept_filter: ConceptFilter | None = None,
        mock_provider: GraphProvider | None = None,
        max_mb: int = 100,
        use_mocked_features: bool = False,
    ) -> None:
        self.model: DifficultyModel = model or RuleBaselineModel()
        self.concept_filter: ConceptFilter = concept_filter or MockConceptFilter()
        self.mock_provider: GraphProvider = mock_provider or MockC1GraphProvider()
        self.max_mb = max_mb
        self.use_mocked_features = use_mocked_features

    def info(self) -> DifficultyInfo:
        return DifficultyInfo(
            model=self.model.info,
            graph_provider=self.mock_provider.name,
            concept_filter=self.concept_filter.name,
            features=list(FEATURE_SPECS),
            todo=list(PHASE1_TODO),
        )

    def run(
        self, data: bytes, filename: str, c1_graph: C1GraphResponse | None = None
    ) -> DifficultyResponse:
        materials = parse_materials(data, filename, max_mb=self.max_mb)
        provider: GraphProvider = (
            PayloadGraphProvider(c1_graph) if c1_graph is not None else self.mock_provider
        )
        ctx = provider.build(materials)
        view = GraphView(ctx.response)
        concepts = [
            c
            for c in build_concepts(ctx, materials, self.concept_filter)
            if c.filter_status == "kept"
        ]
        slides_by_index = {s.index: s for s in materials.slides}

        all_features = []
        fusion_inputs = []
        for c in concepts:
            features = compute_all_features(
                c,
                all_concepts=concepts,
                slides_by_index=slides_by_index,
                structure_available=materials.structure_available,
                metrics=view.metrics(c.id),
                graph_has_prerequisite_edges=view.has_prerequisite_edges,
                graph_source=ctx.source,
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
        outputs = self.model.predict(fusion_inputs)

        predictions: list[ConceptPrediction] = []
        for c, features, out in zip(concepts, all_features, outputs, strict=True):
            warnings: list[str] = []
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
                    features=features,
                    feature_summary=summarize_features(features),
                    status=status,
                    explanation=list(out.explanation) if out else [],
                    warnings=warnings,
                )
            )

        mock_graph = ctx.source.kind == "mock"
        warnings = list(materials.warnings) + list(view.warnings)
        if mock_graph:
            warnings.append(MOCK_GRAPH_WARNING)
        if not view.has_prerequisite_edges:
            warnings.append(NO_PREREQ_WARNING)

        return DifficultyResponse(
            lecture_id=c1_graph.lecture_id if c1_graph is not None else None,
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
            stages=self._stages(materials, ctx.source, view, concepts, all_features, predictions),
            model=self.model.info,
            predictions=predictions,
            todo=list(PHASE1_TODO),
            warnings=warnings,
        )

    def _stages(self, materials, source: InputSource, view, concepts, all_features, predictions):
        mock_graph = source.kind == "mock"
        graph_detail = source.detail
        if view.warnings:
            graph_detail += "; " + "; ".join(view.warnings)
        aligned = sum(1 for c in concepts if c.alignment_status is AlignmentStatus.ALIGNED)
        concept_detail = f"{len(concepts)} concepts, {aligned} aligned, "
        concept_detail += f"{len(concepts) - aligned} not found in materials"
        mock_filter = isinstance(self.concept_filter, MockConceptFilter)
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
