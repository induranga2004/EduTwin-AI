"""C1 HTTP routes (mounted by the app under ``/api/c1``)."""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from edusync_c1.catchup import LateJoinerCatchupEngine, LateJoinerCatchupPack
from edusync_c1.comparison import DualGraphComparator, LectureComparisonReport
from edusync_c1.concepts import extract_concepts
from edusync_c1.documents import combine_page_text, extract_document_pages
from edusync_c1.graph import build_knowledge_graph
from edusync_c1.schemas import (
    GraphResponse,
    HealthResponse,
    KnowledgeGraph,
    Page,
    TranscribeResponse,
    UploadResponse,
)
from edusync_c1.semantic import SemanticMatcher
from edusync_c1.storage import processed_dir, recordings_dir, uploads_dir
from edusync_c1.timeline import build_timeline_compatibility, normalize_transcription_result
from edusync_c1.transcription import Transcriber, get_transcriber
from edusync_c1.validation import (
    UploadError,
    document_limit_bytes,
    read_upload,
    recording_limit_bytes,
    validate_document,
    validate_recording,
)
from edusync_core.settings import CoreSettings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache
def get_matcher() -> SemanticMatcher:
    """FastAPI dependency: one lazily-loading matcher per process."""
    return SemanticMatcher()


SettingsDep = Annotated[CoreSettings, Depends(get_settings)]
TranscriberDep = Annotated[Transcriber, Depends(get_transcriber)]
MatcherDep = Annotated[SemanticMatcher, Depends(get_matcher)]
FileDep = Annotated[UploadFile, File()]
DocumentFileDep = Annotated[UploadFile, File()]


async def _read_document(file: UploadFile, settings: CoreSettings) -> tuple[bytes, list[Page], str]:
    """Read, validate and extract a lecture document; 422 if it has no text."""
    content = await read_upload(file, document_limit_bytes(settings.max_upload_mb))
    pages = extract_document_pages(file.filename or "", content)
    text = combine_page_text(pages)
    if not text:
        raise UploadError(422, "No extractable text was found in the document.")
    return content, pages, text


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse()


@router.post("/lectures/upload", response_model=UploadResponse)
async def upload_lecture(settings: SettingsDep, file: FileDep) -> UploadResponse:
    """Upload a PDF or PPTX and return extracted concepts and page text."""
    try:
        content = await read_upload(file, document_limit_bytes(settings.max_upload_mb))
        extension = validate_document(file.filename, content)
        pages = extract_document_pages(file.filename or "", content)
        full_text = combine_page_text(pages)
        if not full_text:
            raise UploadError(422, "No extractable text was found in the document.")

        lecture_id = str(uuid4())
        (uploads_dir(settings) / f"{lecture_id}{extension}").write_bytes(content)
        concepts = extract_concepts(full_text)
        return UploadResponse(
            lecture_id=lecture_id,
            filename=file.filename,
            page_count=len(pages),
            extracted_character_count=len(full_text),
            concept_count=len(concepts),
            concepts=concepts,
            pages=pages,
            message="Document processed successfully.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(500, "Could not process the uploaded document.") from exc
    finally:
        await file.close()


@router.post("/lectures/graph", response_model=GraphResponse)
async def generate_lecture_graph(settings: SettingsDep, file: FileDep) -> GraphResponse:
    """Generate a candidate knowledge graph from a PDF or PPTX."""
    try:
        _, pages, full_text = await _read_document(file, settings)
        concepts = extract_concepts(full_text)
        return GraphResponse(
            lecture_id=str(uuid4()),
            filename=file.filename,
            page_count=len(pages),
            concept_count=len(concepts),
            concepts=concepts,
            graph=build_knowledge_graph(pages, concepts),
            message="Candidate knowledge graph generated successfully.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Knowledge graph generation failed")
        raise HTTPException(500, "Could not generate the knowledge graph.") from exc
    finally:
        await file.close()


@router.post("/lectures/demo-graph", response_model=GraphResponse)
async def generate_demo_graph(
    settings: SettingsDep,
    name: Annotated[str, Query(description="Sample file name")] = "Cyber_Resilience_Energy_Sector.pptx",
) -> GraphResponse:
    """Generate a prerequisite knowledge graph directly from a bundled sample presentation."""
    candidate_paths = [
        Path(__file__).resolve().parent / "samples" / name,
        Path(__file__).resolve().parents[2] / "data" / "samples" / name,
        uploads_dir(settings) / name,
    ]
    target: Path | None = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            target = p
            break

    if target is None:
        raise HTTPException(404, f"Sample file '{name}' not found.")

    try:
        content = target.read_bytes()
        pages = extract_document_pages(target.name, content)
        full_text = combine_page_text(pages)
        concepts = extract_concepts(full_text)
        return GraphResponse(
            lecture_id=str(uuid4()),
            filename=target.name,
            page_count=len(pages),
            concept_count=len(concepts),
            concepts=concepts,
            graph=build_knowledge_graph(pages, concepts),
            message="Demo knowledge graph generated successfully.",
        )
    except Exception as exc:
        logger.exception("Demo graph generation failed")
        raise HTTPException(500, "Could not generate the demo knowledge graph.") from exc


@router.post("/lectures/transcribe", response_model=TranscribeResponse)
async def transcribe_lecture(
    settings: SettingsDep,
    transcriber: TranscriberDep,
    matcher: MatcherDep,
    file: FileDep,
    document_file: FileDep,
    language: Annotated[str, Query(min_length=2, max_length=10)] = "en",
) -> TranscribeResponse:
    """Transcribe a recording and build a timestamped graph timeline against a document."""
    try:
        recording = await read_upload(file, recording_limit_bytes(settings.max_upload_mb))
        extension = validate_recording(file.filename)
        _, pages, document_text = await _read_document(document_file, settings)

        recording_id = str(uuid4())
        recording_path = recordings_dir(settings) / f"{recording_id}{extension}"
        recording_path.write_bytes(recording)

        raw = await run_in_threadpool(transcriber.transcribe, recording_path, language)
        transcript = normalize_transcription_result(raw)
        if not transcript.segments:
            raise UploadError(
                422,
                "Speech recognition returned no segments. Check the recording.",
            )

        concepts = extract_concepts(document_text)
        response = TranscribeResponse(
            recording_id=recording_id,
            filename=file.filename,
            document_filename=document_file.filename,
            language=transcript.language or language,
            segment_count=len(transcript.segments),
            transcript=transcript.text,
            segments=transcript.segments,
            concept_count=len(concepts),
            concepts=concepts,
            graph=build_knowledge_graph(pages, concepts),
            graph_timeline=await run_in_threadpool(
                build_timeline_compatibility, transcript, pages, matcher
            ),
            message="Recording transcribed and timestamped graph timeline generated successfully.",
        )
        (processed_dir(settings) / f"{recording_id}_transcript.json").write_text(
            json.dumps(response.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Transcription or graph timeline generation failed")
        raise HTTPException(
            500, "Could not transcribe the recording or generate its graph."
        ) from exc
    finally:
        await file.close()
        await document_file.close()


def _load_processed_recording(settings: CoreSettings, recording_id: str) -> dict[str, Any]:
    """Load previously transcribed session data from disk."""
    file_path = processed_dir(settings) / f"{recording_id}_transcript.json"
    if not file_path.is_file():
        raise HTTPException(404, f"Recording '{recording_id}' was not found.")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(500, f"Could not load recording data for '{recording_id}'.") from exc


@router.get("/lectures/{recording_id}/alerts", response_model=LectureComparisonReport)
def get_lecture_alerts(recording_id: str, settings: SettingsDep) -> LectureComparisonReport:
    """Retrieve real-time prerequisite skip alerts and pedagogical alignment for a recording."""
    data = _load_processed_recording(settings, recording_id)
    kg = KnowledgeGraph(**data["graph"])
    comparator = DualGraphComparator(kg)

    label_to_id = {n.label.casefold(): n.id for n in kg.nodes} | {n.id.casefold(): n.id for n in kg.nodes}
    timeline_data = data.get("graph_timeline", {}).get("timeline", [])
    step_mentions = [
        (
            float(step.get("start_seconds", 0)),
            [label_to_id.get(c.casefold(), c.casefold()) for c in step.get("matched_concepts", [])],
        )
        for step in timeline_data
    ]
    return comparator.compare_timeline_steps(step_mentions, lecture_id=recording_id)


@router.get("/lectures/{recording_id}/catchup", response_model=LateJoinerCatchupPack)
def get_lecture_catchup(
    recording_id: str,
    settings: SettingsDep,
    join_seconds: Annotated[float, Query(ge=0.0)] = 300.0,
) -> LateJoinerCatchupPack:
    """Generate a 5-minute topologically ordered catch-up pack for a late-joining student."""
    data = _load_processed_recording(settings, recording_id)
    kg = KnowledgeGraph(**data["graph"])
    engine = LateJoinerCatchupEngine(kg)

    label_to_id = {n.label.casefold(): n.id for n in kg.nodes} | {n.id.casefold(): n.id for n in kg.nodes}
    timeline_data = data.get("graph_timeline", {}).get("timeline", [])
    delivered_mentions = [
        (
            float(step.get("start_seconds", 0)),
            [label_to_id.get(c.casefold(), c.casefold()) for c in step.get("matched_concepts", [])],
            step.get("transcript_text", ""),
        )
        for step in timeline_data
    ]
    return engine.generate_catchup_pack(delivered_mentions, join_seconds, lecture_id=recording_id)


@router.post("/lectures/alerts", response_model=LectureComparisonReport)
async def evaluate_lecture_alerts(
    settings: SettingsDep,
    matcher: MatcherDep,
    transcriber: TranscriberDep,
    file: FileDep,
    document_file: FileDep,
    language: Annotated[str, Query(min_length=2, max_length=10)] = "en",
) -> LectureComparisonReport:
    """Directly evaluate skip alerts from an uploaded document and recording."""
    transcribe_res = await transcribe_lecture(
        settings, transcriber, matcher, file, document_file, language
    )
    comparator = DualGraphComparator(transcribe_res.graph)
    step_mentions = [
        (step.start_seconds, [c.casefold() for c in step.matched_concepts])
        for step in transcribe_res.graph_timeline.timeline
    ]
    return comparator.compare_timeline_steps(
        step_mentions, lecture_id=transcribe_res.recording_id
    )


@router.post("/lectures/catchup", response_model=LateJoinerCatchupPack)
async def generate_lecture_catchup(
    settings: SettingsDep,
    matcher: MatcherDep,
    transcriber: TranscriberDep,
    file: FileDep,
    document_file: FileDep,
    join_seconds: Annotated[float, Query(ge=0.0)] = 300.0,
    language: Annotated[str, Query(min_length=2, max_length=10)] = "en",
) -> LateJoinerCatchupPack:
    """Directly generate a 5-minute catch-up pack from an uploaded document and recording."""
    transcribe_res = await transcribe_lecture(
        settings, transcriber, matcher, file, document_file, language
    )
    engine = LateJoinerCatchupEngine(transcribe_res.graph)
    delivered_mentions = [
        (step.start_seconds, [c.casefold() for c in step.matched_concepts], step.transcript_text)
        for step in transcribe_res.graph_timeline.timeline
    ]
    return engine.generate_catchup_pack(
        delivered_mentions, join_seconds, lecture_id=transcribe_res.recording_id
    )


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("", response_class=HTMLResponse)
def get_viva_dashboard() -> HTMLResponse:
    """Serve the interactive C1 Viva Visual Live Demonstration Dashboard."""
    template_path = Path(__file__).resolve().parent / "templates" / "viva_dashboard.html"
    if not template_path.is_file():
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))

