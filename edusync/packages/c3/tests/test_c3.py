import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c3 import COMPONENT
from edusync_c3.relevance import RelevanceLabel, classify, keyword_overlap_score

CONCEPTS = ["binary search", "time complexity"]  # 4 terms


@pytest.mark.parametrize(
    ("text", "concepts", "expected"),
    [
        ("", CONCEPTS, 0.0),
        ("anything", [], 0.0),
        ("anything", ["  "], 0.0),
        ("Binary search halves the range", CONCEPTS, 0.5),
        ("binary search time complexity", CONCEPTS, 1.0),
        ("what is for lunch", CONCEPTS, 0.0),
    ],
)
def test_score(text, concepts, expected):
    assert keyword_overlap_score(text, concepts) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("binary search is fast", RelevanceLabel.RELEVANT),
        ("the search was long", RelevanceLabel.RELEVANT),  # 1/4 = 0.25
        ("lunch is at noon", RelevanceLabel.UNRELATED),
    ],
)
def test_classify(text, expected):
    assert classify(text, CONCEPTS) == expected


def test_classify_uncertain():
    many = [f"term{i}" for i in range(10)]
    assert classify("term1", many, relevant_threshold=0.5) == RelevanceLabel.UNCERTAIN


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(COMPONENT.router)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"component": "c3", "status": "ok"}


def test_classify_endpoint(client):
    r = client.post("/classify", json={"text": "binary search", "concepts": CONCEPTS})
    assert r.status_code == 200
    assert r.json()["label"] == "relevant"
    assert r.json()["score"] == pytest.approx(0.5)
