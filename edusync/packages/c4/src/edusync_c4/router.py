"""C4 HTTP routes."""

from fastapi import APIRouter

from edusync_c4.evidence import reconcile
from edusync_c4.schemas import ReconcileRequest, ReconcileResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"component": "c4", "status": "ok"}


@router.post("/reconcile")
def reconcile_evidence(req: ReconcileRequest) -> ReconcileResponse:
    return ReconcileResponse(flags=reconcile(req.speech, req.visual, req.window_seconds))
