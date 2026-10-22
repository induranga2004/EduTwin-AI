"""C4 HTTP routes."""

from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from fastapi.responses import HTMLResponse

from edusync_c4.c2_mock import DeterministicDifficultyPredictor, StaticComprehensionEvaluator
from edusync_c4.context import ContextConfig
from edusync_c4.evidence import reconcile
from edusync_c4.grouping import group_timeline
from edusync_c4.interpreter import TemplateContextualInterpreter, analyze_groups
from edusync_c4.mock_data import MOCK_COMPREHENSION, MOCK_CONCEPTS, MOCK_SLIDE_TEXT, MOCK_TRANSCRIPT
from edusync_c4.report import SlideView, build_report_html
from edusync_c4.repository import AnnotationRepository
from edusync_c4.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AnnotationGroup,
    AnnotationTimeline,
    CatchUpItem,
    CatchUpResponse,
    EventImportRequest,
    EventImportResponse,
    GroupingConfig,
    NormalizedAnnotation,
    ReconcileRequest,
    ReconcileResponse,
    SlideInfo,
    SlideSnapshot,
    SlideText,
    SlideUpsert,
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

    speech = req.speech if req.speech is not None else MOCK_TRANSCRIPT
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
            "speech": req.speech is None,
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
    return HTMLResponse(
        build_report_html(lecture_id, views, at),
        headers={"Content-Security-Policy": REPORT_CSP},
    )
