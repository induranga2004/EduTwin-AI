import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT
from edusync_c4.annotations import parse_annotation_text
from edusync_c4.evidence import merge_evidence, reconcile
from edusync_core import ConceptRef, EvidenceItem, EvidenceSource, LectureTimestamp


def ev(cid, source, seconds, text=""):
    return EvidenceItem(
        concept=ConceptRef(id=cid, label=cid),
        source=source,
        timestamp=LectureTimestamp(meeting_id="m", seconds=seconds),
        text=text,
    )


def test_merge_groups_sorts_dedupes():
    items = [
        ev("a", EvidenceSource.SLIDE, 50, "x"),
        ev("a", EvidenceSource.SLIDE, 50, "x"),
        ev("a", EvidenceSource.SPEECH, 10, "y"),
        ev("b", EvidenceSource.POLL, 5),
    ]
    merged = merge_evidence(items)
    assert [i.timestamp.seconds for i in merged["a"]] == [10, 50]
    assert len(merged["b"]) == 1
    assert merge_evidence([]) == {}


def test_reconcile_flags_visual_only():
    speech = [ev("a", EvidenceSource.SPEECH, 100)]
    visual = [
        ev("a", EvidenceSource.SLIDE, 150),  # covered by speech
        ev("b", EvidenceSource.WHITEBOARD, 300),
        ev("b", EvidenceSource.SLIDE, 400),  # same concept, flagged once
        ev("c", EvidenceSource.SLIDE, 20),
        ev("d", EvidenceSource.POLL, 10),  # not visual
    ]
    flags = reconcile(speech, visual)
    assert [f.concept.id for f in flags] == ["c", "b"]
    assert all(f.component == "c4" and f.severity == "review" for f in flags)


def test_reconcile_window():
    speech = [ev("a", EvidenceSource.SPEECH, 0)]
    visual = [ev("a", EvidenceSource.SLIDE, 200)]
    assert len(reconcile(speech, visual)) == 1
    assert reconcile(speech, visual, window_seconds=300) == []


SHAPE = {"props": {"text": "  F = ma "}}


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        (SHAPE, "F = ma"),
        (json.dumps(SHAPE), "F = ma"),
        (
            {"props": {"richText": {"content": [{"content": [{"type": "text", "text": "Hi"}]}]}}},
            "Hi",
        ),
        ({"props": {"text": "", "richText": {"content": [{"text": "A"}, {"text": "B"}]}}}, "AB"),
        ({"props": {"document": {"content": [{"text": "doc"}]}}}, "doc"),
        ({"props": {"richText": "plain"}}, "plain"),
        ({"props": {"text": " "}}, None),
        ({"props": {}}, None),
        ({"props": "x"}, None),
        ({}, None),
        ("not json", None),
        ("[1, 2]", None),
        (None, None),
    ],
)
def test_parse_annotation_text(info, expected):
    assert parse_annotation_text(info) == expected


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(COMPONENT.router)
    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"component": "c4", "status": "ok"}


def test_reconcile_endpoint(client):
    item = ev("b", EvidenceSource.SLIDE, 30).model_dump(mode="json")
    r = client.post("/reconcile", json={"speech": [], "visual": [item]})
    assert r.status_code == 200
    assert r.json()["flags"][0]["concept"]["id"] == "b"
