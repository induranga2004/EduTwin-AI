"""Unit tests for the app factory using fake components."""

import types
from typing import Any

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from edusync_api import __version__, main
from edusync_api.dependencies import get_app_settings
from edusync_api.main import create_app, load_default_components
from edusync_core import Component, CoreSettings


def fake(name: str) -> Component:
    router = APIRouter()

    @router.get("/ping")
    def ping() -> dict[str, str]:
        return {"who": name}

    return Component(name=name, title=name.upper(), description=f"{name} desc", router=router)


def settings(**kw: Any) -> CoreSettings:
    return CoreSettings(_env_file=None, **kw)


def client(**kw: Any) -> TestClient:
    return TestClient(create_app(settings(**kw), [fake("a"), fake("b")]))


def test_health_unconfigured() -> None:
    body = client().get("/health").json()
    assert body == {
        "status": "ok",
        "version": __version__,
        "components": ["a", "b"],
        "bbb": "unconfigured",
    }


@pytest.mark.parametrize("kw", [{"bbb_url": "http://x"}, {"hasura_url": "http://h"}])
def test_health_configured(kw: dict[str, str]) -> None:
    assert client(**kw).get("/health").json()["bbb"] == "configured"


def test_components_listing() -> None:
    data = client().get("/api/components").json()
    assert data[0] == {"name": "a", "title": "A", "description": "a desc", "prefix": "/api/a"}
    assert [d["name"] for d in data] == ["a", "b"]


def test_router_mounting() -> None:
    c = client()
    assert c.get("/api/a/ping").json() == {"who": "a"}
    assert c.get("/api/b/ping").json() == {"who": "b"}


def test_duplicate_name() -> None:
    with pytest.raises(ValueError, match="Duplicate component name"):
        create_app(settings(), [fake("a"), fake("a")])


def test_duplicate_prefix() -> None:
    class Same(Component):
        @property
        def prefix(self) -> str:
            return "/api/same"

    a = Same("x", "X", "", APIRouter())
    b = Same("y", "Y", "", APIRouter())
    with pytest.raises(ValueError, match="Duplicate component prefix"):
        create_app(settings(), [a, b])


def test_cors_header() -> None:
    r = client(cors_origins=["http://ui.test"]).get("/health", headers={"Origin": "http://ui.test"})
    assert r.headers["access-control-allow-origin"] == "http://ui.test"


def test_state() -> None:
    s = settings(bbb_url="http://x")
    app = create_app(s, [fake("a")])
    assert app.state.settings is s
    assert app.title == "EduSync AI"
    assert [c.name for c in app.state.components] == ["a"]


def test_get_app_settings_dependency() -> None:
    app = create_app(settings(bbb_secret="sec"), [])

    @app.get("/_secret")
    def secret(cfg: CoreSettings = Depends(get_app_settings)) -> dict[str, str]:  # noqa: B008
        return {"s": cfg.bbb_secret}

    assert TestClient(app).get("/_secret").json() == {"s": "sec"}


def test_load_default_components(monkeypatch: pytest.MonkeyPatch) -> None:
    mods = {f"edusync_c{i}": types.SimpleNamespace(COMPONENT=fake(f"c{i}")) for i in range(1, 5)}
    monkeypatch.setattr(main.importlib, "import_module", lambda name: mods[name])
    assert [c.name for c in load_default_components()] == ["c1", "c2", "c3", "c4"]
    app = create_app(settings())
    assert [c.name for c in app.state.components] == ["c1", "c2", "c3", "c4"]


def test_main_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    import edusync_api.__main__ as entry

    calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("EDUSYNC_HOST", "0.0.0.0")
    entry.main()
    assert calls == [(("edusync_api.asgi:app",), {"host": "0.0.0.0", "port": 8100})]
