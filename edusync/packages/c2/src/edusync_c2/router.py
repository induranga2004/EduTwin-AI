"""C2 HTTP routes."""

import json
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from edusync_c2.comprehension import flag_for_clarification
from edusync_c2.difficulty.materials import MaterialsError
from edusync_c2.difficulty.pipeline import DifficultyPipeline
from edusync_c2.difficulty.schemas import C1GraphResponse, DifficultyInfo, DifficultyResponse
from edusync_c2.schemas import ComprehensionRequest, ComprehensionResponse
from edusync_core.settings import CoreSettings, get_settings

router = APIRouter()

MAX_UPLOAD_MB_CAP = 100


@lru_cache
def get_pipeline() -> DifficultyPipeline:
    """FastAPI dependency: the default pipeline (rule baseline, mock graph provider)."""
    return DifficultyPipeline()


PipelineDep = Annotated[DifficultyPipeline, Depends(get_pipeline)]
SettingsDep = Annotated[CoreSettings, Depends(get_settings)]


def _parse_graph(raw: str | None) -> C1GraphResponse | None:
    if raw is None or not raw.strip():
        return None
    try:
        return C1GraphResponse.model_validate(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"c1_graph is not valid JSON: {exc.msg}") from exc
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
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
    graph = _parse_graph(c1_graph)
    limit_mb = min(settings.max_upload_mb, MAX_UPLOAD_MB_CAP)
    try:
        data = await file.read(limit_mb * 1024 * 1024 + 1)
    finally:
        await file.close()
    if len(data) > limit_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {limit_mb} MB limit")
    try:
        return await run_in_threadpool(pipeline.run, data, file.filename or "", graph)
    except MaterialsError as exc:
        raise HTTPException(exc.status_code, exc.message) from exc


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
