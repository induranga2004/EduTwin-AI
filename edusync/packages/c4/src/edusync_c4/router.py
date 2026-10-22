"""C4 HTTP routes."""

import re
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

from edusync_c4.c2_mock import DeterministicDifficultyPredictor, StaticComprehensionEvaluator
from edusync_c4.context import ContextConfig
from edusync_c4.diagram import build_diagrams
from edusync_c4.evidence import reconcile
from edusync_c4.grouping import group_timeline
from edusync_c4.interpreter import TemplateContextualInterpreter, analyze_groups
from edusync_c4.local_explain import METHOD as LOCAL_METHOD
from edusync_c4.local_explain import explain_slide, speech_on_slide
from edusync_c4.mock_data import MOCK_COMPREHENSION, MOCK_CONCEPTS, MOCK_SLIDE_TEXT, MOCK_TRANSCRIPT
from edusync_c4.report import SlideView, build_report_html, fullest_time
from edusync_c4.repository import AnnotationRepository
from edusync_c4.schemas import (
    SYNC_ALIASES,
    AnalyzeRequest,
    AnalyzeResponse,
    AnnotationGroup,
    AnnotationTimeline,
    CatchUpItem,
    CatchUpResponse,
    Diagram,
    EventImportRequest,
    EventImportResponse,
    ExplainResponse,
    GroupingConfig,
    LectureSummary,
    NormalizedAnnotation,
    ReconcileRequest,
    ReconcileResponse,
    SlideInfo,
    SlideSeen,
    SlideSnapshot,
    SlideText,
    SlideUpsert,
    SlideVisit,
    SlideWord,
    SlideWordsUpsert,
    SpeechInfo,
    SpeechResponse,
    SpeechUpload,
)
from edusync_c4.timeline import reconstruct_snapshot
from edusync_core import CoreSettings, get_settings

router = APIRouter()

REPORT_CSP = "default-src 'none'; img-src data:; style-src 'unsafe-inline'"
CATCHUP_NOTE = (
    "Items are ordered by time only. Concept candidates and quiz signals come from mock data; "
    "prerequisite ordering needs the C1 concept graph and is not implemented."
)
HIDDEN = "hidden"


def get_repository(
    settings: Annotated[CoreSettings, Depends(get_settings)],
) -> AnnotationRepository:
    """Repository under `<data_dir>/c4/annotations.db`; tests override this dependency."""
    return AnnotationRepository(settings.data_dir / "c4" / "annotations.db")


RepoDep = Annotated[AnnotationRepository, Depends(get_repository)]


def _scrub(ann: NormalizedAnnotation, include_raw: bool) -> NormalizedAnnotation:
    """Hide the raw payload and author id unless explicitly requested."""
    if include_raw:
        return ann
    return ann.model_copy(update={"raw_payload": {}, "shape_state": {}, "user_id": HIDDEN})


def _timeline_or_404(
    repo: AnnotationRepository, lecture_id: str, slide_id: str
) -> AnnotationTimeline:
    timeline = repo.timeline(lecture_id, slide_id)
    if timeline is None:
        raise HTTPException(404, f"No events for lecture {lecture_id!r} slide {slide_id!r}.")
    return timeline


@router.get("/health")
def health() -> dict[str, str]:
    return {"component": "c4", "status": "ok"}


@router.post("/reconcile")
def reconcile_evidence(req: ReconcileRequest) -> ReconcileResponse:
    return ReconcileResponse(flags=reconcile(req.speech, req.visual, req.window_seconds))


@router.post("/events/import")
def import_events(req: EventImportRequest, repo: RepoDep) -> EventImportResponse:
    return repo.import_events(
        req.events, req.lecture_id, req.slide_id, req.time_unit, req.lecture_start_ms
    )


@router.get("/lectures/{lecture_id}/events")
def list_events(
    lecture_id: str,
    repo: RepoDep,
    slide_id: str | None = None,
    include_raw: bool = False,
) -> list[NormalizedAnnotation]:
    if not repo.lecture_exists(lecture_id):
        raise HTTPException(404, f"Unknown lecture {lecture_id!r}.")
    slides = [slide_id] if slide_id is not None else repo.slide_ids(lecture_id)
    out: list[NormalizedAnnotation] = []
    for slide in slides:
        timeline = repo.timeline(lecture_id, slide)
        if timeline is not None:
            out.extend(_scrub(a, include_raw) for a in [*timeline.events, *timeline.unplaced])
    return out


@router.get("/lectures/{lecture_id}/slides/{slide_id:path}/snapshot")
def snapshot(
    lecture_id: str,
    slide_id: str,
    repo: RepoDep,
    timestamp: Annotated[float, Query(ge=0)],
    include_raw: bool = False,
) -> SlideSnapshot:
    snap = reconstruct_snapshot(_timeline_or_404(repo, lecture_id, slide_id), timestamp)
    snap.annotations = [_scrub(a, include_raw) for a in snap.annotations]
    return snap


@router.post("/lectures/{lecture_id}/slides/{slide_id:path}/group")
def compute_groups(
    lecture_id: str,
    slide_id: str,
    repo: RepoDep,
    config: Annotated[GroupingConfig | None, Body()] = None,
) -> list[AnnotationGroup]:
    cfg = config or GroupingConfig()
    groups = group_timeline(_timeline_or_404(repo, lecture_id, slide_id), cfg)
    repo.replace_groups(lecture_id, slide_id, groups, cfg)
    return groups


@router.get("/lectures/{lecture_id}/groups")
def list_groups(
    lecture_id: str, repo: RepoDep, slide_id: str | None = None
) -> list[AnnotationGroup]:
    if not repo.lecture_exists(lecture_id):
        raise HTTPException(404, f"Unknown lecture {lecture_id!r}.")
    return repo.load_groups(lecture_id, slide_id)


@router.post("/lectures/{lecture_id}/slides/{slide_id:path}/analyze")
def analyze(
    lecture_id: str,
    slide_id: str,
    repo: RepoDep,
    req: Annotated[AnalyzeRequest | None, Body()] = None,
) -> AnalyzeResponse:
    req = req or AnalyzeRequest()
    timeline = _timeline_or_404(repo, lecture_id, slide_id)
    groups = repo.load_groups(lecture_id, slide_id)
    if not groups:
        cfg = GroupingConfig()
        groups = group_timeline(timeline, cfg)
        repo.replace_groups(lecture_id, slide_id, groups, cfg)

    stored = repo.load_speech(lecture_id) if req.speech is None else None
    speech = req.speech if req.speech is not None else (stored[1] if stored else MOCK_TRANSCRIPT)
    slide_text = SlideText(
        slide_id=slide_id,
        text=req.slide_text if req.slide_text is not None else MOCK_SLIDE_TEXT.text,
    )
    concepts = req.concepts if req.concepts is not None else MOCK_CONCEPTS
    interpreter = (
        TemplateContextualInterpreter(
            DeterministicDifficultyPredictor(), StaticComprehensionEvaluator(MOCK_COMPREHENSION)
        )
        if req.use_mock_c2
        else TemplateContextualInterpreter()
    )
    analyses = analyze_groups(
        timeline,
        groups,
        speech,
        slide_text,
        concepts,
        config=ContextConfig(
            speech_pad_seconds=req.speech_pad_seconds,
            speech_offset_seconds=req.speech_offset_seconds,
        ),
        interpreter=interpreter,
    )
    repo.save_analyses(lecture_id, analyses)
    return AnalyzeResponse(
        lecture_id=lecture_id,
        slide_id=slide_id,
        inputs_are_mock={
            "speech": req.speech is None and stored is None,
            "slide_text": req.slide_text is None,
            "concepts": req.concepts is None,
            "c2": req.use_mock_c2,
        },
        analyses=analyses,
    )


@router.get("/lectures/{lecture_id}/catchup")
def catchup(
    lecture_id: str,
    repo: RepoDep,
    since: Annotated[float, Query(ge=0)] = 0.0,
) -> CatchUpResponse:
    if not repo.lecture_exists(lecture_id):
        raise HTTPException(404, f"Unknown lecture {lecture_id!r}.")
    items = [
        CatchUpItem(
            slide_id=slide,
            group_id=a.context.group_id,
            start_seconds=start,
            end_seconds=end,
            category=a.interpretation.category,
            summary=a.interpretation.summary,
            needs_attention=a.interpretation.needs_attention,
            cited_evidence_ids=a.interpretation.cited_evidence_ids,
            limitations=a.interpretation.limitations,
        )
        for slide, start, end, a in repo.load_analyses(lecture_id, since)
    ]
    return CatchUpResponse(
        lecture_id=lecture_id, since_seconds=since, items=items, note=CATCHUP_NOTE
    )


# Specific "/slides/{slide}/<suffix>" routes are declared above; this one is the bare slide id.
@router.get("/lectures/{lecture_id}/slides")
def list_slides(lecture_id: str, repo: RepoDep) -> list[SlideInfo]:
    return repo.list_slides(lecture_id)


@router.get("/lectures/{lecture_id}/slides/{slide_id:path}/svg")
def slide_svg(lecture_id: str, slide_id: str, repo: RepoDep) -> Response:
    svg = repo.get_slide_svg(lecture_id, slide_id)
    if not svg:
        raise HTTPException(404, f"No slide image for lecture {lecture_id!r} slide {slide_id!r}.")
    # Served as a download-safe image; scripts never run when SVG is loaded via <img>.
    return Response(svg, media_type="image/svg+xml", headers={"Content-Security-Policy": "sandbox"})


# The "/words" routes must come BEFORE the bare slide route: its {slide_id:path} would otherwise
# swallow "<slide>/words" as a slide id.
@router.put("/lectures/{lecture_id}/slides/{slide_id:path}/words")
def put_slide_words(
    lecture_id: str, slide_id: str, body: SlideWordsUpsert, repo: RepoDep
) -> dict[str, int]:
    """Store the position of every word on a slide (from the slide PDF), in whiteboard units."""
    return {"words": repo.save_slide_words(lecture_id, slide_id, body.words)}


@router.get("/lectures/{lecture_id}/slides/{slide_id:path}/words")
def get_slide_words(lecture_id: str, slide_id: str, repo: RepoDep) -> list[SlideWord]:
    return repo.get_slide_words(lecture_id, slide_id)


@router.post("/lectures/{lecture_id}/slides/{slide_id:path}/seen")
def post_slide_seen(
    lecture_id: str, slide_id: str, body: SlideSeen, repo: RepoDep
) -> dict[str, bool]:
    """Note that this slide is on screen right now (called by the capture script every ~2 s)."""
    repo.record_slide_seen(lecture_id, slide_id, body.seconds)
    return {"recorded": True}


@router.get("/lectures/{lecture_id}/visits")
def get_visits(lecture_id: str, repo: RepoDep) -> list[SlideVisit]:
    """When each slide was on screen, oldest first."""
    return sorted(
        (
            SlideVisit(slide_id=slide_id, start_seconds=start, end_seconds=end)
            for slide_id, spans in repo.get_slide_visits(lecture_id).items()
            for start, end in spans
        ),
        key=lambda v: (v.start_seconds, v.slide_id),
    )


@router.put("/lectures/{lecture_id}/slides/{slide_id:path}")
def put_slide(lecture_id: str, slide_id: str, slide: SlideUpsert, repo: RepoDep) -> SlideInfo:
    return repo.save_slide(lecture_id, slide_id, slide)


@router.delete("/lectures/{lecture_id}")
def delete_lecture(lecture_id: str, repo: RepoDep) -> dict[str, int]:
    """Remove all stored data for a lecture (for example rehearsal data)."""
    return repo.delete_lecture(lecture_id)


@router.get("/lectures/{lecture_id}/report", response_class=HTMLResponse)
def report(
    lecture_id: str,
    repo: RepoDep,
    at: Annotated[float | None, Query(ge=0, description="Show the board at this second")] = None,
) -> HTMLResponse:
    """Printable lecture report: slides with the lecturer's drawings, times and text."""
    if not repo.lecture_exists(lecture_id):
        raise HTTPException(404, f"Unknown lecture {lecture_id!r}.")
    infos = {s.slide_id: s for s in repo.list_slides(lecture_id)}
    views: list[SlideView] = []
    for slide_id in repo.slide_ids(lecture_id):
        timeline = repo.timeline(lecture_id, slide_id)
        if timeline is None:
            continue
        views.append(
            SlideView(
                slide_id=slide_id,
                info=infos.get(slide_id),
                svg=repo.get_slide_svg(lecture_id, slide_id),
                timeline=timeline,
                groups=group_timeline(timeline, GroupingConfig()),
            )
        )
    views.sort(
        key=lambda v: (v.info.number if v.info and v.info.number is not None else 10**6, v.slide_id)
    )
    stored = repo.load_speech(lecture_id)
    speech = stored[1] if stored else None
    saved = {a.context.group_id: a for _, _, _, a in repo.load_analyses(lecture_id)}
    visits = repo.get_slide_visits(lecture_id)
    for view in views:  # stored explanations win (for example AI ones); else explain live
        view.analyses = {g.group_id: saved[g.group_id] for g in view.groups if g.group_id in saved}
        missing = [g for g in view.groups if g.group_id not in view.analyses]
        slide_text = view.info.text if view.info else None
        view.words = repo.get_slide_words(lecture_id, view.slide_id)
        view.speech = speech_on_slide(speech, visits.get(view.slide_id))
        for a in explain_slide(view.timeline, missing, view.speech, slide_text, words=view.words):
            view.analyses[a.context.group_id] = a
    return HTMLResponse(
        build_report_html(lecture_id, views, at, speech=speech),
        headers={"Content-Security-Policy": REPORT_CSP},
    )


@router.get("/lectures/{lecture_id}/slides/{slide_id:path}/diagrams")
def slide_diagrams(
    lecture_id: str,
    slide_id: str,
    repo: RepoDep,
    timestamp: Annotated[float | None, Query(ge=0)] = None,
) -> list[Diagram]:
    """Boxes joined by arrows on a slide, at `timestamp` (default: when the board was fullest)."""
    timeline = _timeline_or_404(repo, lecture_id, slide_id)
    when = timestamp if timestamp is not None else fullest_time(timeline)
    return build_diagrams(timeline.get_active_annotations(when))


def _find_sync_time(segments, word: str) -> float | None:
    """Recording time at which `word` (or a common mis-hearing of it) was first spoken."""
    words = SYNC_ALIASES.get(word.lower(), (word.lower(),))
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(w) for w in words) + r")(?!\w)", re.I)
    for seg in sorted(segments, key=lambda g: g.start):
        if pattern.search(seg.text):
            return seg.start
    return None


@router.get("/lectures")
def list_lectures(repo: RepoDep) -> list[LectureSummary]:
    """Every stored lecture, newest first (use the id in the other endpoints)."""
    return repo.list_lectures()


@router.put("/lectures/{lecture_id}/speech")
def put_speech(lecture_id: str, upload: SpeechUpload, repo: RepoDep) -> SpeechInfo:
    """Store the lecturer's transcript and line it up with the meeting clock."""
    sync_shape: str | None = None
    if upload.offset_seconds is not None:
        offset, method = upload.offset_seconds, "manual"
    elif upload.sync_word:
        spoken = _find_sync_time(upload.segments, upload.sync_word)
        if spoken is None:
            raise HTTPException(
                422, f"The word {upload.sync_word!r} was not found in the transcript."
            )
        first = repo.first_annotation(lecture_id)
        if first is None:
            raise HTTPException(422, f"No annotations stored for lecture {lecture_id!r} yet.")
        sync_shape, drawn = first
        offset, method = drawn - spoken, "sync_word"
    else:
        offset, method = 0.0, "none"
    return repo.save_speech(lecture_id, upload.segments, offset, method, upload.source, sync_shape)


@router.get("/lectures/{lecture_id}/speech")
def get_speech(lecture_id: str, repo: RepoDep) -> SpeechResponse:
    """The stored transcript with times on the meeting clock."""
    stored = repo.load_speech(lecture_id)
    if stored is None:
        raise HTTPException(404, f"No speech stored for lecture {lecture_id!r}.")
    info, segments = stored
    return SpeechResponse(**info.model_dump(), segments=segments)


@router.post("/lectures/{lecture_id}/explain")
def explain_lecture(lecture_id: str, repo: RepoDep) -> ExplainResponse:
    """(Re)build and store the "What it means" summaries for every slide of a lecture.

    Uses real inputs only: the stored speech and slide text, and no mock data. The report builds
    the same summaries on the fly, so this is only needed to store them (for example to compare
    interpreters later).
    """
    if not repo.lecture_exists(lecture_id):
        raise HTTPException(404, f"Unknown lecture {lecture_id!r}.")
    stored = repo.load_speech(lecture_id)
    speech = stored[1] if stored else None
    infos = {s.slide_id: s for s in repo.list_slides(lecture_id)}
    visits = repo.get_slide_visits(lecture_id)
    slides = groups_total = 0
    for slide_id in repo.slide_ids(lecture_id):
        timeline = repo.timeline(lecture_id, slide_id)
        if timeline is None:
            continue
        groups = group_timeline(timeline, GroupingConfig())
        repo.replace_groups(lecture_id, slide_id, groups, GroupingConfig())
        info = infos.get(slide_id)
        words = repo.get_slide_words(lecture_id, slide_id)
        slide_speech = speech_on_slide(speech, visits.get(slide_id))
        repo.save_analyses(
            lecture_id,
            explain_slide(timeline, groups, slide_speech, info.text if info else None, words=words),
        )
        slides += 1
        groups_total += len(groups)
    return ExplainResponse(
        lecture_id=lecture_id,
        method=LOCAL_METHOD,
        slide_count=slides,
        group_count=groups_total,
        used_speech=bool(speech),
        used_slide_text=any(i.text for i in infos.values()),
    )
