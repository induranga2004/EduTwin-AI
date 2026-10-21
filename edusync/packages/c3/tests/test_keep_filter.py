import importlib

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c3 import COMPONENT, KeepFilter, ModelUnavailableError, split_sentences

router_module = importlib.import_module("edusync_c3.router")  # `edusync_c3.router` is the APIRouter


def test_split_basic_and_whitespace():
    text = "Good morning everyone.  Today we start!\nIs it clear? Yes."
    assert split_sentences(text) == [
        "Good morning everyone.",
        "Today we start!",
        "Is it clear?",
        "Yes.",
    ]
    assert split_sentences("   ") == []
    assert split_sentences("no end punctuation") == ["no end punctuation"]


def test_split_keeps_abbreviations_and_initials_together():
    text = (
        "Sri Lanka has its own Act, Act No. 36 of 2003. "
        "The book J.K. Rowling wrote is protected, e.g. the exact wording. "
        "Dr. Smith disagrees."
    )
    assert split_sentences(text) == [
        "Sri Lanka has its own Act, Act No. 36 of 2003.",
        "The book J.K. Rowling wrote is protected, e.g. the exact wording.",
        "Dr. Smith disagrees.",
    ]


class Enc:
    def __init__(self, ids):
        self.ids = ids
        self.attention_mask = [1] * len(ids)


class FakeTokenizer:
    def encode_batch(self, texts):
        return [Enc(list(range(1, len(t.split()) + 1))) for t in texts]


class FakeSession:
    """KEEP logit grows with the word count: 1 word -> p~0.12, 3 words -> p~0.88."""

    def run(self, _outputs, feeds):
        n_words = feeds["attention_mask"].sum(axis=1)
        return [np.stack([np.zeros(len(n_words)), (n_words - 2.0) * 2.0], axis=1)]


def make(threshold=0.4):
    return KeepFilter(FakeSession(), FakeTokenizer(), ["0", "1"], 8, threshold=threshold)


def test_scores_and_filter_follow_threshold():
    f = make()
    scored = f.score(["hi", "so this is teaching", "  "])  # blank sentences are dropped
    assert [s.text for s in scored] == ["hi", "so this is teaching"]
    assert scored[0].p_keep < 0.4 < scored[1].p_keep
    assert [s.keep for s in scored] == [False, True]
    assert f.keep_only(["hi", "so this is teaching"]) == ["so this is teaching"]
    assert make(0.99).keep_only(["two words"]) == []


def test_filter_rejects_bad_setup():
    with pytest.raises(ValueError, match="KEEP label"):
        KeepFilter(FakeSession(), FakeTokenizer(), ["a", "b"], 8)
    with pytest.raises(ValueError, match="threshold"):
        make(1.0)


def test_missing_model_dir_is_reported(tmp_path):
    with pytest.raises(ModelUnavailableError, match="EDUSYNC_C3_MODEL_DIR"):
        KeepFilter.from_dir(tmp_path)


@pytest.fixture
def client(monkeypatch):
    router_module.get_keep_filter.cache_clear()
    monkeypatch.setattr(router_module, "get_keep_filter", lambda: make())
    app = FastAPI()
    app.include_router(COMPONENT.router)
    return TestClient(app)


def test_filter_endpoint_splits_text_and_keeps_teaching(client):
    r = client.post("/filter", json={"text": "Hello. This sentence teaches something. Ok."})
    assert r.status_code == 200
    body = r.json()
    assert body["kept"] == ["This sentence teaches something."]
    assert (body["total"], body["kept_count"], body["threshold"]) == (3, 1, 0.4)
    assert body["items"] is None


def test_filter_endpoint_options(client):
    payload = {"sentences": ["Hello", "This is teaching"], "include_dropped": True}
    body = client.post("/filter", json={**payload, "threshold": 0.95}).json()
    assert body["kept"] == []
    assert body["threshold"] == 0.95
    assert [i["keep"] for i in body["items"]] == [False, False]


def test_filter_endpoint_validation(client):
    assert client.post("/filter", json={}).status_code == 422
    assert client.post("/filter", json={"text": "a", "sentences": ["a"]}).status_code == 422
    assert client.post("/filter", json={"text": "a", "threshold": 1.5}).status_code == 422


def test_filter_endpoint_503_without_model(monkeypatch, tmp_path):
    router_module.get_keep_filter.cache_clear()
    monkeypatch.setenv("EDUSYNC_C3_MODEL_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(COMPONENT.router)
    r = TestClient(app).post("/filter", json={"text": "Some sentence."})
    assert r.status_code == 503
    assert "EDUSYNC_C3_MODEL_DIR" in r.json()["detail"]
