import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c2 import COMPONENT
from edusync_c2.comprehension import flag_for_clarification
from edusync_c2.difficulty import lexical_difficulty
from edusync_core import ConceptRef, FlagSeverity, LectureTimestamp

CONCEPT = ConceptRef(id="c1", label="Recursion")
TS = LectureTimestamp(meeting_id="m1", seconds=120)


def test_difficulty_empty_is_zero():
    assert lexical_difficulty("") == 0.0
    assert lexical_difficulty("123 !!!") == 0.0


def test_difficulty_orders_simple_below_technical():
    simple = lexical_difficulty("The cat sat on the mat. It was a big red hat.")
    hard = lexical_difficulty(
        "Asymptotic complexity characterisation necessitates considering "
        "multidimensional probabilistic distributions across heterogeneous architectures."
    )
    assert 0.0 <= simple < hard <= 1.0


@pytest.mark.parametrize(
    ("correct", "total", "diff", "expected"),
    [
        (1, 2, 0.0, None),  # too few responses
        (0, 0, 0.0, None),
        (8, 10, 0.0, None),
        (6, 10, 0.0, None),  # exactly at threshold
        (5, 10, 0.0, FlagSeverity.REVIEW),
        (2, 10, 0.0, FlagSeverity.URGENT),
        (6, 10, 1.0, FlagSeverity.REVIEW),  # difficulty raises threshold to 0.7
        (3, 10, 0.0, FlagSeverity.REVIEW),  # 0.3 is not < 0.3
    ],
)
def test_flag_for_clarification(correct, total, diff, expected):
    flag = flag_for_clarification(CONCEPT, correct, total, TS, predicted_difficulty=diff)
    if expected is None:
        assert flag is None
    else:
        assert flag is not None
        assert flag.severity == expected
        assert flag.component == "c2"
        assert flag.concept == CONCEPT


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(COMPONENT.router)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"component": "c2", "status": "ok"}
    assert COMPONENT.prefix == "/api/c2"


def test_difficulty_endpoint(client):
    r = client.post("/difficulty", json={"text": "Hello world."})
    assert r.status_code == 200
    assert 0 <= r.json()["score"] <= 1


def test_evaluate_endpoint(client):
    body = {
        "concept": {"id": "c1", "label": "Recursion"},
        "correct": 1,
        "total": 10,
        "timestamp": {"meeting_id": "m1", "seconds": 5},
    }
    r = client.post("/comprehension/evaluate", json=body)
    assert r.json()["flag"]["severity"] == "urgent"
    ok = client.post("/comprehension/evaluate", json={**body, "correct": 9})
    assert ok.json()["flag"] is None


def test_evaluate_rejects_correct_above_total(client):
    body = {
        "concept": {"id": "c1", "label": "R"},
        "correct": 5,
        "total": 3,
        "timestamp": {"meeting_id": "m1", "seconds": 5},
    }
    assert client.post("/comprehension/evaluate", json=body).status_code == 422
