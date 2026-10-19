"""C1 HTTP routes (mounted by the app under ``/api/c1``)."""

import json
import logging
from functools import lru_cache
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from edusync_c1.concepts import extract_concepts
from edusync_c1.documents import combine_page_text, extract_document_pages
from edusync_c1.graph import build_knowledge_graph
from edusync_c1.schemas import (
    GraphResponse,
    HealthResponse,
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
    """Upload a PDF and return extracted concepts and page text."""
    try:
        content = await read_upload(file, document_limit_bytes(settings.max_upload_mb))
        if validate_document(file.filename, content) != ".pdf":
            raise UploadError(400, "This endpoint accepts PDF files only.")
        pages = extract_document_pages(file.filename or "", content)
        full_text = combine_page_text(pages)
        if not full_text:
            raise UploadError(422, "No extractable text was found. The PDF may require OCR.")

        lecture_id = str(uuid4())
        (uploads_dir(settings) / f"{lecture_id}.pdf").write_bytes(content)
        concepts = extract_concepts(full_text)
        return UploadResponse(
            lecture_id=lecture_id,
            filename=file.filename,
            page_count=len(pages),
            extracted_character_count=len(full_text),
            concept_count=len(concepts),
            concepts=concepts,
            pages=pages,
            message="PDF processed successfully.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PDF upload failed")
        raise HTTPException(500, "Could not process the uploaded PDF.") from exc
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
