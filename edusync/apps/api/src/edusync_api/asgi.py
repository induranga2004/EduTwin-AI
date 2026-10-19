"""ASGI entrypoint: ``uvicorn edusync_api.asgi:app``."""

from edusync_api.main import create_app

app = create_app()
