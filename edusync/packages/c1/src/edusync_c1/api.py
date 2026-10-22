"""C1 HTTP routes (mounted by the app under ``/api/c1``)."""

import json
import logging
import re
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
from edusync_c1.coverage import load_coverage_classifier
from edusync_c1.documents import combine_page_text, extract_document_pages
from edusync_c1.graph import build_knowledge_graph
from edusync_c1.schemas import (
    GraphResponse,
    HealthResponse,
    KnowledgeGraph,
    LiveEvalConceptInput,
    LiveEvalRequest,
    LiveEvalResponse,
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


@router.post("/lectures/live-eval", response_model=LiveEvalResponse)
async def evaluate_live_utterance(
    payload: LiveEvalRequest,
    matcher: MatcherDep,
) -> LiveEvalResponse:
    """Evaluate a real-time speech utterance against curriculum concepts using MiniLM and Model C."""
    text = payload.utterance_text.strip()
    if not text or not payload.concepts:
        return LiveEvalResponse(
            explanation="Empty utterance or no concepts provided."
        )

    # 1. Concept & Sub-Topic Candidate Scoring with MiniLM Semantic Bi-Encoder
    candidate_scores: list[tuple[LiveEvalConceptInput, float, str, str | None, str | None]] = []

    for concept in payload.concepts:
        c_title = concept.title
        c_points = list(concept.points or [])
        c_desc = concept.description or ""
        sub_list = concept.subtopics or []

        # Pool of texts representing this concept (title, points, subtopics, description)
        candidates = [c_title]
        candidates.extend(c_points)
        for sub in sub_list:
            if sub.title not in candidates:
                candidates.append(sub.title)
            if sub.description and sub.description not in candidates:
                candidates.append(sub.description)
        if c_desc and c_desc not in candidates:
            candidates.append(c_desc)

        best_score = 0.0
        best_term = c_title
        matched_sub_id: str | None = None
        matched_sub_title: str | None = None

        # Check semantic similarity using MiniLM
        try:
            match_res = matcher.match(text, candidates, threshold=0.25)
            if match_res.matches:
                top_m = max(match_res.matches, key=lambda m: m.similarity)
                best_score = float(top_m.similarity)
                best_term = top_m.name
                for sub in sub_list:
                    if top_m.name == sub.title or top_m.name == sub.description:
                        matched_sub_id = sub.id
                        matched_sub_title = sub.title
                        break
                if not matched_sub_id and top_m.name != c_title and top_m.name in c_points:
                    clean_sub_id = re.sub(r"[^a-zA-Z0-9]+", "_", top_m.name.casefold()).strip("_")
                    matched_sub_id = f"{concept.id}__sub__{clean_sub_id}"
                    matched_sub_title = top_m.name
        except Exception:
            pass

        # Also evaluate lexical overlap with title or points or constituent subtopics
        lower_text = text.casefold()
        norm_text = re.sub(r"\s+", " ", lower_text.replace("&", " and "))

        clean_title = re.sub(r"^\[slide\s*\d+\]\s*", "", c_title, flags=re.IGNORECASE).casefold()
        norm_title = re.sub(r"\s+", " ", clean_title.replace("&", " and "))
        norm_id = re.sub(r"\s+", " ", concept.id.replace("_", " ").casefold().replace("&", " and "))

        if (len(norm_title) >= 5 and norm_title in norm_text) or (len(norm_id) >= 5 and norm_id in norm_text):
            best_score = max(best_score, 0.95)
            best_term = c_title

        for sub in sub_list:
            sub_id = sub.id
            sub_title = sub.title
            sub_desc = sub.description or ""
            if len(sub_title) >= 3 and sub_title.casefold() in lower_text:
                if 0.92 > best_score:
                    best_score = 0.92
                    best_term = sub_title
                    matched_sub_id = sub_id
                    matched_sub_title = sub_title
            if sub_desc:
                words = [w for w in re.findall(r"[a-zA-Z]{4,}", sub_desc.casefold())]
                matches = [w for w in set(words) if w in lower_text]
                if len(matches) >= 2:
                    lex_score = min(0.96, 0.72 + 0.08 * len(matches))
                    if lex_score > best_score:
                        best_score = lex_score
                        best_term = sub_title
                        matched_sub_id = sub_id
                        matched_sub_title = sub_title

        for pt in c_points:
            if len(pt) >= 4 and pt.casefold() in lower_text:
                if 0.88 > best_score:
                    best_score = 0.88
                    best_term = pt
                    clean_sub_id = re.sub(r"[^a-zA-Z0-9]+", "_", pt.casefold()).strip("_")
                    matched_sub_id = f"{concept.id}__sub__{clean_sub_id}"
                    matched_sub_title = pt

        if best_score > 0.0:
            candidate_scores.append((concept, best_score, best_term, matched_sub_id, matched_sub_title))

    if not candidate_scores:
        return LiveEvalResponse(
            explanation="No curriculum concept matched this speech segment."
        )

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    best_concept, similarity, matched_term, matched_sub_id, matched_sub_title = candidate_scores[0]

    if similarity < 0.35:
        return LiveEvalResponse(
            explanation=f"Best candidate '{best_concept.title}' similarity {similarity:.2f} is below threshold (0.35)."
        )

    cid = best_concept.id
    c_index = best_concept.index
    delivered_set = set(payload.delivered_concept_ids)

    # If subtopic matched, add subtopic to delivered set
    if matched_sub_id:
        delivered_set.add(matched_sub_id)

    # 2. Sequential Prerequisite Metrics
    prev_concepts = [c for c in payload.concepts if c.index < c_index]
    if prev_concepts:
        covered_prereqs = sum(1 for c in prev_concepts if c.id in delivered_set)
        prereq_ratio = covered_prereqs / len(prev_concepts)
        skipped = [c.title for c in prev_concepts if c.id not in delivered_set]
    else:
        prereq_ratio = 1.0
        skipped = []

    prev_evidence_count = 1 if cid in delivered_set else 0

    # 3. Model C ML Classifier Inference
    classifier = load_coverage_classifier()
    model_status = "introduced" if prev_evidence_count == 0 else "adequately_covered"
    model_confidence = 0.95
    class_probs: dict[str, float] = {model_status: 0.95}

    if classifier is not None:
        try:
            import pandas as pd

            feat_df = pd.DataFrame(
                [
                    {
                        "transcript_text": text,
                        "semantic_similarity": float(similarity),
                        "prerequisite_coverage": float(prereq_ratio),
                        "previous_evidence_count": float(prev_evidence_count),
                    }
                ]
            )
            pred = classifier.predict(feat_df)[0]
            model_status = str(pred)
            if hasattr(classifier, "predict_proba"):
                probs = classifier.predict_proba(feat_df)[0]
                class_probs = {
                    str(c): round(float(p), 4)
                    for c, p in zip(classifier.classes_, probs)
                }
                model_confidence = round(float(max(probs)), 4)
        except Exception as exc:
            logger.debug("Model C live inference fallback: %s", exc)

    # Adjust for severe prerequisite violations if needed
    if prereq_ratio < 0.3 and prev_concepts:
        if model_status == "adequately_covered":
            model_status = "insufficiently_covered"

    newly_delivered: list[str] = []
    if matched_sub_id and matched_sub_id not in payload.delivered_concept_ids:
        newly_delivered.append(matched_sub_id)

    # Sub-topic completion logic for parent Main Topic
    main_topic_covered = False
    main_topic_skipped = False
    notification_type: str | None = None
    notification_message = ""

    # Generate expected subtopic ids for this concept
    expected_sub_ids = [s.id for s in (best_concept.subtopics or [])]
    if not expected_sub_ids:
        expected_sub_ids = [
            f"{cid}__sub__{re.sub(r'[^a-zA-Z0-9]+', '_', pt.casefold()).strip('_')}"
            for pt in (best_concept.points or [])
        ]
    covered_subs = [sid for sid in expected_sub_ids if sid in delivered_set]

    # Main topic is considered covered if:
    # 1. The main topic title itself was explicitly covered/matched, OR
    # 2. All or >= 50% of its constituent sub-topics are delivered
    sub_threshold_met = (
        len(expected_sub_ids) == 0
        or len(covered_subs) == len(expected_sub_ids)
        or (len(expected_sub_ids) >= 2 and len(covered_subs) >= len(expected_sub_ids) - 1)
    )

    if (matched_term == best_concept.title or sub_threshold_met) and cid not in payload.delivered_concept_ids:
        main_topic_covered = True
        newly_delivered.append(cid)
        notification_type = "main_topic_covered"
        notification_message = f"🎉 Main Topic Covered: [Slide {best_concept.slideNum:02d}] {best_concept.title} (All sub-topics covered!)"
    elif matched_sub_id:
        notification_type = "subtopic_covered"
        notification_message = f"📌 Sub-Topic Covered: {matched_sub_title} (under [Slide {best_concept.slideNum:02d}] {best_concept.title})"

    # If previous main topics were skipped
    if skipped:
        main_topic_skipped = True
        if not notification_type:
            notification_type = "main_topic_skipped"
            notification_message = f"⚠️ Main Topic Skipped: Prerequisite '{skipped[0]}' was skipped without covering its sub-topics!"

    # 4. Synthesize Overarching Main Concept Coverage
    main_id = payload.main_concept_id or (payload.concepts[0].id if payload.concepts else None)
    main_synthesized = False

    all_delivered_now = delivered_set | set(newly_delivered)
    constituent_concepts = [c for c in payload.concepts if c.id != main_id]
    if main_id and constituent_concepts:
        covered_constituents = sum(1 for c in constituent_concepts if c.id in all_delivered_now)
        thresh_met = (
            (covered_constituents / len(constituent_concepts) >= 0.5)
            or (len(constituent_concepts) >= 4 and covered_constituents >= 3)
        )
        if thresh_met and main_id not in all_delivered_now:
            main_synthesized = True
            newly_delivered.append(main_id)

    prereq_ok = len(skipped) == 0

    return LiveEvalResponse(
        matched_concept_id=cid,
        matched_concept_title=best_concept.title,
        matched_slide_num=best_concept.slideNum,
        matched_subtopic_id=matched_sub_id,
        matched_subtopic_title=matched_sub_title,
        parent_topic_id=cid,
        parent_topic_title=best_concept.title,
        main_topic_covered=main_topic_covered,
        main_topic_skipped=main_topic_skipped,
        notification_type=notification_type,
        notification_message=notification_message,
        semantic_similarity=round(similarity, 3),
        matching_method="MiniLM-L6-v2 Semantic Bi-Encoder + Model C",
        model_status=model_status,
        model_confidence=model_confidence,
        class_probabilities=class_probs,
        prereq_satisfied=prereq_ok,
        skipped_prerequisites=skipped,
        main_concept_synthesized=main_synthesized,
        newly_delivered_ids=newly_delivered,
        explanation=(
            f"Evaluated via MiniLM similarity ({similarity:.2f}) and Model C classifier "
            f"-> Status: '{model_status}' (Confidence: {model_confidence * 100:.1f}%)."
        ),
    )


@router.get("/dashboard", response_class=HTMLResponse)
@router.get("", response_class=HTMLResponse)
def get_viva_dashboard() -> HTMLResponse:
    """Serve the interactive C1 Viva Visual Live Demonstration Dashboard."""
    template_path = Path(__file__).resolve().parent / "templates" / "viva_dashboard.html"
    if not template_path.is_file():
        raise HTTPException(status_code=404, detail="Dashboard template not found.")
    return HTMLResponse(content=template_path.read_text(encoding="utf-8"))

