import json
from io import BytesIO

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pptx import Presentation

from edusync_c2 import COMPONENT
from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.synthetic import build_c1_graph, build_pdf, build_pptx, make_deck_spec
from edusync_c2.router import get_pipeline
from edusync_core.settings import CoreSettings, get_settings

SPEC = make_deck_spec(5, 5, "lec-r")
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(COMPONENT.router)
    a.dependency_overrides[get_pipeline] = lambda: DifficultyPipeline()
    return a


@pytest.fixture
def client(app):
    return TestClient(app)


def post(client, data=None, name="deck.pptx", graph=None):
    files = {"file": (name, build_pptx(SPEC) if data is None else data, PPTX_MIME)}
    form = {} if graph is None else {"c1_graph": graph}
    return client.post("/difficulty", files=files, data=form)


def test_with_graph(client):
    r = post(client, graph=json.dumps(build_c1_graph(SPEC, "v1")))
    assert r.status_code == 200
    body = r.json()
    assert body["lecture_id"] == "lec-r"
    assert len(body["predictions"]) == 5
    assert body["inputs"][1]["kind"] == "real"


def test_without_graph(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json()["inputs"][1]["kind"] == "mock"
    assert post(client, graph="  ").json()["inputs"][1]["kind"] == "mock"


def test_pdf(client):
    r = post(client, build_pdf(SPEC), "deck.pdf")
    assert r.status_code == 200
    assert r.json()["file_type"] == "pdf"


def test_wrong_type(client):
    assert post(client, b"hello", "a.txt").status_code == 400


def test_too_large(app, client):
    app.dependency_overrides[get_settings] = lambda: CoreSettings(max_upload_mb=1)
    r = post(client, b"PK" + b"0" * (1024 * 1024 + 10))
    assert r.status_code == 413


def test_invalid_json(client):
    r = post(client, graph="{not json")
    assert r.status_code == 422
    assert "not valid JSON" in r.json()["detail"]


def test_invalid_schema(client):
    r = post(client, graph=json.dumps({"lecture_id": "x"}))
    assert r.status_code == 422
    assert "C1's response shape" in r.json()["detail"]


def test_textless_file(client):
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    buf = BytesIO()
    prs.save(buf)
    assert post(client, buf.getvalue()).status_code == 422


def test_old_text_body_rejected(client):
    assert client.post("/difficulty", json={"text": "Hello world."}).status_code == 422


def test_info(client):
    r = client.get("/difficulty/info")
    assert r.status_code == 200
    assert len(r.json()["features"]) == 24
    assert r.json()["model"]["kind"] == "baseline"


def test_health_and_comprehension_untouched(client):
    assert client.get("/health").json() == {"component": "c2", "status": "ok"}
    body = {
        "concept": {"id": "c1", "label": "R"},
        "correct": 1,
        "total": 10,
        "timestamp": {"meeting_id": "m", "seconds": 5},
    }
    assert client.post("/comprehension/evaluate", json=body).json()["flag"]["severity"] == "urgent"


def test_default_pipeline_dependency():
    get_pipeline.cache_clear()
    assert get_pipeline() is get_pipeline()
    assert isinstance(get_pipeline(), DifficultyPipeline)
