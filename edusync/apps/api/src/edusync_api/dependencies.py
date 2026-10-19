"""FastAPI dependencies."""

from fastapi import Request

from edusync_core import CoreSettings


def get_app_settings(request: Request) -> CoreSettings:
    """Return the settings the app was built with."""
    return request.app.state.settings
