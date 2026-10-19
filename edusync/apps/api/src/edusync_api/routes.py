"""Platform-level routes (health and component discovery)."""

from typing import Any

from fastapi import APIRouter, Request

from edusync_api import __version__

router = APIRouter()


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Liveness probe with mounted components and BBB configuration state."""
    settings = request.app.state.settings
    configured = bool(settings.hasura_url or settings.bbb_url)
    return {
        "status": "ok",
        "version": __version__,
        "components": [c.name for c in request.app.state.components],
        "bbb": "configured" if configured else "unconfigured",
    }


@router.get("/api/components")
def list_components(request: Request) -> list[dict[str, str]]:
    """Describe every mounted component."""
    return [
        {"name": c.name, "title": c.title, "description": c.description, "prefix": c.prefix}
        for c in request.app.state.components
    ]
