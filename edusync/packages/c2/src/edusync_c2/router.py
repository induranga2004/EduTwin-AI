"""C2 HTTP routes."""

import json
from collections.abc import Callable
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from edusync_c2.comprehension import flag_for_clarification
from edusync_c2.difficulty.c1_client import UrllibTransport
from edusync_c2.difficulty.concepts import create_concept_filter
from edusync_c2.difficulty.errors import (
    GraphProviderError,
    ModelArtifactError,
    ModelNotAvailableError,
    SemanticScoringError,
)
from edusync_c2.difficulty.features.semantic import SemanticScorer
from edusync_c2.difficulty.features.semantic_loader import get_default_encoder
from edusync_c2.difficulty.materials import MaterialsError
from edusync_c2.difficulty.models import create_model
from edusync_c2.difficulty.pipeline import SEMANTIC_UNAVAILABLE, DifficultyPipeline
from edusync_c2.difficulty.providers import (
    C1ApiGraphProvider,
    FileGraphProvider,
    GraphProvider,
    MockC1GraphProvider,
    find_mocked_graph,
)
from edusync_c2.difficulty.schemas import C1GraphResponse, DifficultyInfo, DifficultyResponse
from edusync_c2.logger import (
    configure_logging,
    get_logger,
    new_request_id,
    reset_request_id,
    set_request_id,
)
from edusync_c2.schemas import ComprehensionRequest, ComprehensionResponse
from edusync_c2.settings import C2Settings, get_c2_settings
from edusync_core.settings import CoreSettings, get_settings

logger = get_logger(__name__)
router = APIRouter()

MAX_UPLOAD_MB_CAP = 100


def _graph_provider_factory(settings: C2Settings) -> Callable[[bytes, str], GraphProvider] | None:
    """The per-request provider for uploads without a posted graph (None = slide-title mock)."""
    if settings.graph_provider == "file":
        file_provider = FileGraphProvider(settings.graph_file)
        return lambda data, filename: file_provider
    if settings.graph_provider == "c1_api":
        base_url, timeout_s = settings.c1_base_url, settings.c1_timeout_s
        return lambda data, filename: C1ApiGraphProvider(
            data, filename, UrllibTransport(base_url, timeout_s)
        )
    return None


def _semantic_scorer(settings: C2Settings) -> tuple[SemanticScorer | None, str]:
    """The semantic scorer and its status string ("off", "minilm:<sha8>", "minilm:unavailable")."""
    if settings.semantic == "off":
        return None, "off"
    encoder = get_default_encoder(settings.semantic_model_dir)
    if encoder is None:
        return None, SEMANTIC_UNAVAILABLE
    try:
        scorer = SemanticScorer(
            encoder,
            settings.semantic_def_threshold,
            settings.semantic_ex_threshold,
            settings.semantic_max_sentences,
        )
    except SemanticScoringError:
        logger.warning("semantic_unavailable reason=prototype_encoding")
        return None, SEMANTIC_UNAVAILABLE
    return scorer, f"minilm:{encoder.sha}"


@lru_cache
def get_pipeline() -> DifficultyPipeline:
    """FastAPI dependency: the pipeline for the configured model and graph provider.

    Failures raise and are therefore not cached by ``lru_cache``.
    """
    settings = get_c2_settings()
    configure_logging(settings.log_level)
    try:
        model = create_model(settings.model, settings.model_dir)
    except (ModelNotAvailableError, ModelArtifactError) as exc:
        logger.warning("model_unavailable name=%s", settings.model)
        raise HTTPException(501, str(exc)) from exc
    try:
        concept_filter = create_concept_filter(
            settings.concept_filter, settings.concept_filter_extras
        )
    except NotImplementedError as exc:
        logger.warning("concept_filter_unavailable name=%s", settings.concept_filter)
        raise HTTPException(501, str(exc)) from exc
    semantic, semantic_status = _semantic_scorer(settings)
    logger.info("pipeline_ready model=%s", settings.model)
    return DifficultyPipeline(
        model=model,
        concept_filter=concept_filter,
        selected_model=settings.model,
        graph_provider_factory=_graph_provider_factory(settings),
        graph_provider_name=None if settings.graph_provider == "mock" else settings.graph_provider,
        semantic=semantic,
        semantic_status=semantic_status,
    )


PipelineDep = Annotated[DifficultyPipeline, Depends(get_pipeline)]
SettingsDep = Annotated[CoreSettings, Depends(get_settings)]


def _parse_graph(raw: str | None) -> C1GraphResponse | None:
    if raw is None or not raw.strip():
        return None
    try:
        return C1GraphResponse.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        logger.warning("rejected status=422 reason=c1_graph_invalid_json")
        raise HTTPException(422, f"c1_graph is not valid JSON: {exc.msg}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        logger.warning("rejected status=422 reason=c1_graph_shape")
        raise HTTPException(
            422, f"c1_graph does not match C1's response shape: {where}: {first['msg']}"
        ) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"component": "c2", "status": "ok"}


@router.post("/difficulty", response_model=DifficultyResponse)
async def difficulty(
    pipeline: PipelineDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    c1_graph: Annotated[str | None, Form()] = None,
) -> DifficultyResponse:
    """Predict per-concept difficulty from a PPTX/PDF and an optional C1 graph (JSON text)."""
    token = set_request_id(new_request_id())
    try:
        graph = _parse_graph(c1_graph)
        limit_mb = min(settings.max_upload_mb, MAX_UPLOAD_MB_CAP)
        try:
            data = await file.read(limit_mb * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(data) > limit_mb * 1024 * 1024:
            logger.warning("rejected status=413 reason=too_large limit_mb=%d", limit_mb)
            raise HTTPException(413, f"File exceeds the {limit_mb} MB limit")
        try:
            return await run_in_threadpool(pipeline.run, data, file.filename or "", graph)
        except MaterialsError as exc:
            logger.warning("rejected status=%d reason=%s", exc.status_code, exc.message)
            raise HTTPException(exc.status_code, exc.message) from exc
        except GraphProviderError as exc:
            logger.warning("rejected status=%d reason=graph_provider", exc.status_code)
            raise HTTPException(exc.status_code, exc.message) from exc
        except SemanticScoringError as exc:
            logger.warning("rejected status=%d reason=semantic_scorer", exc.status_code)
            raise HTTPException(exc.status_code, exc.message) from exc
        except NotImplementedError as exc:
            logger.warning("rejected status=501 reason=model_not_implemented")
            raise HTTPException(501, str(exc)) from exc
    finally:
        reset_request_id(token)


def run_analysis(*args, **kwargs) -> dict:
    """Layered analysis; imported lazily because it needs numpy (the package imports without)."""
    from edusync_c2.difficulty.scoring.analyze import analyze

    return analyze(*args, **kwargs)


def get_score_artifact() -> object:
    """FastAPI dependency: the loaded score artifact, or 501 with the reason."""
    from edusync_c2.difficulty.scoring.artifact import get_artifact
    from edusync_c2.difficulty.scoring.reference import ScoringError

    settings = get_c2_settings()
    if settings.score_dir is None:
        raise HTTPException(501, "Layered analysis is not configured: set EDUSYNC_C2_SCORE_DIR")
    try:
        return get_artifact(settings.score_dir, model_dir=settings.semantic_model_dir)
    except ScoringError as exc:
        logger.warning("score_artifact_unavailable reason=%s", exc)
        raise HTTPException(501, f"Score artifact cannot be used: {exc}") from exc


ArtifactDep = Annotated[object, Depends(get_score_artifact)]


@lru_cache
def _analysis_parts() -> tuple[object, object, str]:
    """Concept filter, semantic scorer and its status for layered analysis (built once)."""
    settings = get_c2_settings()
    try:
        concept_filter = create_concept_filter(
            settings.concept_filter, settings.concept_filter_extras
        )
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    semantic, status = _semantic_scorer(settings)
    return concept_filter, semantic, status


def _demo_graph_provider(
    settings: C2Settings, filename: str
) -> Callable[[bytes, str], GraphProvider]:
    path = find_mocked_graph(filename, settings.demo_graph_dir)
    if path is None:  # unknown lecture: fall back to the normal provider (slide-title mock graph)
        return _graph_provider_factory(settings) or (lambda data, name: MockC1GraphProvider())
    return lambda data, name: FileGraphProvider(path)


@router.post("/difficulty/analyze")
async def difficulty_analyze(
    artifact: ArtifactDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File()],
    graph: Annotated[str | None, Form()] = None,
    use_demo_graph: Annotated[bool, Form()] = False,
) -> dict:
    """Layered difficulty analysis: every layer plus ranked hotspots, as one JSON document."""
    token = set_request_id(new_request_id())
    try:
        posted = _parse_graph(graph)
        c2 = get_c2_settings()
        limit_mb = min(settings.max_upload_mb, MAX_UPLOAD_MB_CAP)
        try:
            data = await file.read(limit_mb * 1024 * 1024 + 1)
        finally:
            await file.close()
        if len(data) > limit_mb * 1024 * 1024:
            logger.warning("rejected status=413 reason=too_large limit_mb=%d", limit_mb)
            raise HTTPException(413, f"File exceeds the {limit_mb} MB limit")
        name = file.filename or ""
        factory = None
        if posted is None and use_demo_graph:
            factory = _demo_graph_provider(c2, name)
        elif posted is None:
            factory = _graph_provider_factory(c2)
        concept_filter, semantic, status = _analysis_parts()
        try:
            return await run_in_threadpool(
                lambda: run_analysis(
                    data,
                    name,
                    artifact=artifact,
                    graph=posted,
                    provider_factory=factory,
                    concept_filter=concept_filter,
                    semantic=semantic,
                    semantic_status=status,
                    max_mb=limit_mb,
                )
            )
        except MaterialsError as exc:
            logger.warning("rejected status=%d reason=%s", exc.status_code, exc.message)
            raise HTTPException(exc.status_code, exc.message) from exc
        except GraphProviderError as exc:
            logger.warning("rejected status=%d reason=graph_provider", exc.status_code)
            raise HTTPException(exc.status_code, exc.message) from exc
        except SemanticScoringError as exc:
            logger.warning("rejected status=%d reason=semantic_scorer", exc.status_code)
            raise HTTPException(exc.status_code, exc.message) from exc
    finally:
        reset_request_id(token)


def _ui_text(name: str) -> str | None:
    try:
        return resources.files("edusync_c2").joinpath("ui", name).read_text(encoding="utf-8")
    except OSError:
        return None


@router.get("/demo", response_class=HTMLResponse)
def demo_page() -> HTMLResponse:
    html = _ui_text("index.html")
    if html is None:
        return HTMLResponse("The demo page is not installed", status_code=503)
    return HTMLResponse(html)


@router.get("/demo/sample")
def demo_sample() -> JSONResponse:
    text = _ui_text("sample_analysis.json")
    if text is None:
        raise HTTPException(503, "The sample analysis is not installed")
    return JSONResponse(json.loads(text))


@router.get("/demo/evaluation")
def demo_evaluation() -> JSONResponse:
    score_dir = get_c2_settings().score_dir
    path = None if score_dir is None else Path(score_dir) / "demo_evaluation.json"
    if path is None or not path.is_file():
        raise HTTPException(404, "No demo evaluation is available")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@router.get("/difficulty/info", response_model=DifficultyInfo)
def difficulty_info(pipeline: PipelineDep) -> DifficultyInfo:
    return pipeline.info()


@router.post("/comprehension/evaluate")
def evaluate(req: ComprehensionRequest) -> ComprehensionResponse:
    flag = flag_for_clarification(
        req.concept,
        req.correct,
        req.total,
        req.timestamp,
        predicted_difficulty=req.predicted_difficulty,
        threshold=req.threshold,
        min_responses=req.min_responses,
    )
    return ComprehensionResponse(flag=flag)
