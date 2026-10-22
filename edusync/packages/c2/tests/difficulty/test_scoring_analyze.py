import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from scoring_helpers import deck, fake_artifact, write_artifact_dir

from edusync_c2 import COMPONENT
from edusync_c2.difficulty.errors import SemanticScoringError
from edusync_c2.difficulty.providers import FileGraphProvider
from edusync_c2.difficulty.schemas import C1Graph, C1GraphResponse
from edusync_c2.difficulty.scoring.analyze import analyze
from edusync_c2.difficulty.scoring.artifact import clear_artifact_cache
from edusync_c2.settings import C2Settings, get_c2_settings
from edusync_core.settings import CoreSettings, get_settings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contract_keys import contract_errors

RT = importlib.import_module("edusync_c2.router")
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
SPEC, DATA, GRAPH = deck()


@pytest.fixture(scope="module")
def artifact():
    return fake_artifact()


def run(artifact, **kw):
    graph = kw.pop("graph", C1GraphResponse.model_validate(GRAPH))
    return analyze(DATA, "deck.pptx", artifact=artifact, graph=graph, **kw)


def test_contract_and_determinism(artifact):
    doc = run(artifact)
    assert contract_errors(doc) == []
    assert doc == run(artifact)
    assert doc["meta"]["graph"]["kind"] == "real"
    assert doc["meta"]["concept_count"] == len(doc["final"]["concepts"]) == 6
    assert doc["final"]["hotspots"][0]["rank"] == 1
    assert "provisional" in doc["disclaimer"]
    # no slide text leaks: only titles, concept labels and counts
    parse = doc["layers"]["parse"]
    assert set(parse["slides"][0]) == {"index", "title", "words"}
    for g in "ABCD":
        assert doc["layers"][g]["feature_directions"]


def test_graph_sources(artifact, tmp_path):
    mock = analyze(DATA, "deck.pptx", artifact=artifact)
    assert mock["meta"]["graph"]["kind"] == "mock"
    path = tmp_path / "g.json"
    path.write_text(json.dumps({**GRAPH, "curated": True}), encoding="utf-8")
    provider = FileGraphProvider(path)
    cur = analyze(DATA, "deck.pptx", artifact=artifact, provider_factory=lambda d, n: provider)
    assert cur["meta"]["graph"]["kind"] == "curated"
    low_conf = {c["concept_id"]: c["confidence"] for c in mock["final"]["concepts"]}
    assert all(0 <= v <= 1 for v in low_conf.values())


def test_missing_models_and_failures():
    none = run(fake_artifact(t1=False, t2=False))
    assert contract_errors(none) == []
    assert none["meta"]["text_models"] == {"t1": False, "t2": False}
    assert all(r["t1"] is None and r["t2"] is None for r in none["layers"]["text"]["per_concept"])
    one = run(fake_artifact(t2=False))
    assert one["layers"]["text"]["per_concept"][0]["agreement"] is None
    broken = fake_artifact()
    broken.t1.fail = True
    doc = run(broken)
    assert any("T1 failed" in w for w in doc["warnings"])
    assert doc["layers"]["text"]["per_concept"][0]["t1"] is None


def test_agreement_buckets(artifact):
    doc = run(artifact)
    kinds = {r["agreement"] for r in doc["layers"]["text"]["per_concept"]}
    assert kinds <= {"agree", "partial", "disagree"}
    from edusync_c2.difficulty.scoring.analyze import _agreement

    assert [_agreement(0.1, 0.2), _agreement(0.1, 0.35), _agreement(0, 0.9)] == [
        "agree",
        "partial",
        "disagree",
    ]


def test_no_concepts(artifact):
    empty = C1GraphResponse(graph=C1Graph())
    doc = run(artifact, graph=empty)
    assert doc["meta"]["concept_count"] == 0
    assert any("No concepts" in w for w in doc["warnings"])
    assert doc["layers"]["A"]["summary"]["available_share"] == 0.0
    assert doc["final"]["hotspots"] or doc["final"]["slide_scores"] is not None


# ---------------------------------------------------------------------------------- router


@pytest.fixture
def env(tmp_path, monkeypatch, artifact):
    RT._analysis_parts.cache_clear()
    clear_artifact_cache()
    holder = {"settings": C2Settings(_env_file=None)}
    monkeypatch.setattr(RT, "get_c2_settings", lambda: holder["settings"])
    monkeypatch.setattr(RT, "_semantic_scorer", lambda s: (None, "off"))
    app = FastAPI()
    app.include_router(COMPONENT.router)
    app.dependency_overrides[RT.get_score_artifact] = lambda: artifact
    holder["app"] = app
    holder["tmp"] = tmp_path
    yield holder
    RT._analysis_parts.cache_clear()
    clear_artifact_cache()


def post(env, data=DATA, name="deck.pptx", **form):
    return TestClient(env["app"]).post(
        "/difficulty/analyze", files={"file": (name, data, PPTX_MIME)}, data=form
    )


def test_endpoint_ok_variants(env):
    r = post(env)
    assert r.status_code == 200
    assert contract_errors(r.json()) == []
    assert r.json()["meta"]["graph"]["kind"] == "mock"
    assert post(env, graph=json.dumps(GRAPH)).json()["meta"]["graph"]["kind"] == "real"
    env["settings"] = C2Settings(_env_file=None, graph_provider="file")
    assert post(env).status_code == 200


def test_demo_graph(env):
    directory = env["tmp"] / "graphs"
    directory.mkdir()
    env["settings"] = C2Settings(_env_file=None, demo_graph_dir=directory)
    assert post(env, use_demo_graph="true").status_code == 404
    (directory / "default.graph.json").write_text(
        json.dumps({**GRAPH, "curated": True}), encoding="utf-8"
    )
    assert post(env, use_demo_graph="true").json()["meta"]["graph"]["kind"] == "curated"
    (directory / "deck.graph.json").write_text("{", encoding="utf-8")
    assert post(env, use_demo_graph="true").status_code == 500  # the deck's own graph wins
    env["settings"] = C2Settings(_env_file=None)
    assert post(env, use_demo_graph="true").status_code == 404


def test_endpoint_errors(env, monkeypatch):
    assert post(env, data=b"hi", name="a.txt").status_code == 400
    assert post(env, graph="{").status_code == 422
    env["app"].dependency_overrides[get_settings] = lambda: CoreSettings(max_upload_mb=1)
    assert post(env, data=b"0" * (1024 * 1024 + 1)).status_code == 413
    env["app"].dependency_overrides.pop(get_settings)

    def raises(exc):
        def inner(*a, **k):
            raise exc

        return inner

    monkeypatch.setattr(RT, "run_analysis", raises(SemanticScoringError()))
    assert post(env).status_code == 503
    from edusync_c2.difficulty.errors import GraphProviderError

    monkeypatch.setattr(RT, "run_analysis", raises(GraphProviderError(502, "x")))
    assert post(env).status_code == 502
    RT._analysis_parts.cache_clear()
    monkeypatch.setattr(RT, "create_concept_filter", raises(NotImplementedError("nf")))
    assert post(env).status_code == 501


def test_artifact_dependency(tmp_path, monkeypatch):
    clear_artifact_cache()
    holder = {"s": C2Settings(_env_file=None)}
    monkeypatch.setattr(RT, "get_c2_settings", lambda: holder["s"])
    app = FastAPI()
    app.include_router(COMPONENT.router)
    client = TestClient(app)
    files = {"file": ("deck.pptx", DATA, PPTX_MIME)}
    r = client.post("/difficulty/analyze", files=files)
    assert r.status_code == 501
    assert "EDUSYNC_C2_SCORE_DIR" in r.json()["detail"]
    holder["s"] = C2Settings(_env_file=None, score_dir=tmp_path)
    assert client.post("/difficulty/analyze", files=files).status_code == 501
    write_artifact_dir(tmp_path)
    (tmp_path / "t1_int8.onnx").unlink()
    holder["s"] = C2Settings(_env_file=None, score_dir=tmp_path)
    ok = client.post("/difficulty/analyze", files=files)
    assert ok.status_code == 200
    assert ok.json()["meta"]["text_models"] == {"t1": False, "t2": False}
    clear_artifact_cache()


def test_demo_routes(env, monkeypatch):
    client = TestClient(env["app"])
    page = client.get("/demo")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    sample = client.get("/demo/sample")
    assert sample.status_code == 200
    assert contract_errors(sample.json()) == []
    assert RT._ui_text("no-such-file.html") is None
    monkeypatch.setattr(RT, "_ui_text", lambda name: None)
    assert client.get("/demo").status_code == 503
    assert client.get("/demo/sample").status_code == 503


def test_demo_evaluation(env):
    client = TestClient(env["app"])
    assert client.get("/demo/evaluation").status_code == 404
    env["settings"] = C2Settings(_env_file=None, score_dir=env["tmp"])
    assert client.get("/demo/evaluation").status_code == 404
    (env["tmp"] / "demo_evaluation.json").write_text('{"n": 3}', encoding="utf-8")
    assert client.get("/demo/evaluation").json() == {"n": 3}


def test_settings_fields(monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_SCORE_DIR", "")
    monkeypatch.setenv("EDUSYNC_C2_DEMO_GRAPH_DIR", "some/dir")
    s = C2Settings(_env_file=None)
    assert s.score_dir is None
    assert str(s.demo_graph_dir).replace("\\", "/") == "some/dir"
    assert get_c2_settings() is not None
