"""Application factory."""

import importlib
from collections.abc import Sequence

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edusync_api import __version__
from edusync_api.routes import router as platform_router
from edusync_core import Component, CoreSettings, get_settings

DEFAULT_COMPONENTS = ("c1", "c2", "c3", "c4")


def load_default_components() -> list[Component]:
    """Import ``edusync_c1``..``edusync_c4`` and collect their ``COMPONENT``."""
    return [importlib.import_module(f"edusync_{n}").COMPONENT for n in DEFAULT_COMPONENTS]


def _check_unique(components: Sequence[Component]) -> None:
    names: set[str] = set()
    prefixes: set[str] = set()
    for c in components:
        if c.name in names:
            raise ValueError(f"Duplicate component name: {c.name!r}")
        if c.prefix in prefixes:
            raise ValueError(f"Duplicate component prefix: {c.prefix!r}")
        names.add(c.name)
        prefixes.add(c.prefix)


def create_app(
    settings: CoreSettings | None = None,
    components: Sequence[Component] | None = None,
) -> FastAPI:
    """Build the EduSync app, mounting each component router under ``/api/<name>``."""
    settings = settings or get_settings()
    comps = list(components) if components is not None else load_default_components()
    _check_unique(comps)

    app = FastAPI(title="EduSync AI", version=__version__)
    app.state.settings = settings
    app.state.components = comps
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(platform_router)
    for c in comps:
        app.include_router(c.router, prefix=c.prefix, tags=[c.name])
    return app
