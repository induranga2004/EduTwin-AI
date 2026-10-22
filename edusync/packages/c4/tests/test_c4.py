import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT
from edusync_c4.annotations import parse_annotation_text
from edusync_c4.evidence import merge_evidence, reconcile
from edusync_c4.normalizer import (
    apply_delta,
    normalize_event,
    normalize_events,
)
from edusync_c4.schemas import BoundingBox, ShapeType
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


# --- Phase 1: Foundation & Normalization Tests ---


def test_extended_parse_annotation_text():
    # Sticky note
    assert parse_annotation_text({"props": {"text": "Note content"}}) == "Note content"
    # Frame container title
    assert parse_annotation_text({"props": {"name": "Container Box"}}) == "Container Box"
    # Poll questionText
    assert (
        parse_annotation_text({"props": {"questionText": "What is momentum?"}})
        == "What is momentum?"
    )
    # Poll question alternative key
    assert parse_annotation_text({"props": {"question": "Is F=ma valid?"}}) == "Is F=ma valid?"


def test_normalize_tldraw_draw_stroke():
    raw = {
        "id": "shape:draw1",
        "type": "draw",
        "x": 100.0,
        "y": 150.0,
        "meetingId": "lecture-101",
        "pageId": "slide-3",
        "timestamp": 45.2,
        "props": {"segments": [{"points": [[0, 0, 0.4], [10, 20, 0.6], [50, 60, 0.8]]}]},
        "userId": "user-prof",
    }
    norm = normalize_event(raw)
    assert norm.shape_id == "shape:draw1"
    assert norm.shape_type == ShapeType.DRAW
    assert norm.lecture_id == "lecture-101"
    assert norm.slide_id == "slide-3"
    assert norm.timestamp_seconds == 45.2
    assert norm.bounds == BoundingBox(x=100.0, y=150.0, width=50.0, height=60.0)
    assert len(norm.stroke_points) == 3
    assert norm.stroke_points[1].x == 110.0
    assert norm.stroke_points[1].y == 170.0
    assert norm.stroke_points[1].pressure == 0.6
    assert norm.user_id == "user-prof"
    assert norm.is_deleted is False
    assert norm.raw_payload == raw


def test_normalize_tldraw_text_and_note():
    # Text shape with growY
    text_raw = {
        "id": "shape:text1",
        "type": "text",
        "x": 200.0,
        "y": 300.0,
        "props": {"text": "Newton's Second Law", "w": 180.0, "h": 40.0, "growY": 10.0},
    }
    norm_text = normalize_event(text_raw, default_lecture_id="m1", default_slide_id="p1")
    assert norm_text.shape_type == ShapeType.TEXT
    assert norm_text.text_content == "Newton's Second Law"
    assert norm_text.bounds == BoundingBox(x=200.0, y=300.0, width=180.0, height=50.0)

    # Note shape default bounds
    note_raw = {
        "id": "shape:note1",
        "type": "note",
        "x": 50.0,
        "y": 80.0,
        "props": {"text": "Office hours on Thursday"},
    }
    norm_note = normalize_event(note_raw)
    assert norm_note.shape_type == ShapeType.NOTE
    assert norm_note.text_content == "Office hours on Thursday"
    assert norm_note.bounds == BoundingBox(x=50.0, y=80.0, width=200.0, height=200.0)


def test_normalize_tldraw_geo_and_arrow():
    # Geo with label
    geo_raw = {
        "id": "shape:geo1",
        "type": "geo",
        "x": 30.0,
        "y": 40.0,
        "props": {"geo": "rectangle", "w": 120.0, "h": 80.0, "text": "Subsystem A"},
    }
    norm_geo = normalize_event(geo_raw)
    assert norm_geo.shape_type == ShapeType.GEO
    assert norm_geo.text_content == "Subsystem A"
    assert norm_geo.bounds == BoundingBox(x=30.0, y=40.0, width=120.0, height=80.0)

    # Arrow connector
    arrow_raw = {
        "id": "shape:arrow1",
        "type": "arrow",
        "x": 50.0,
        "y": 50.0,
        "props": {"start": {"x": 0.0, "y": 0.0}, "end": {"x": 100.0, "y": 80.0}},
    }
    norm_arrow = normalize_event(arrow_raw)
    assert norm_arrow.shape_type == ShapeType.ARROW
    assert norm_arrow.bounds == BoundingBox(x=50.0, y=50.0, width=100.0, height=80.0)


def test_apply_delta_state_merge():
    base = {
        "id": "shape:1",
        "type": "text",
        "x": 100,
        "y": 100,
        "props": {"text": "Initial Text", "w": 120, "h": 40},
    }
    delta = {"x": 150, "props": {"w": 200}}
    merged = apply_delta(base, delta)
    assert merged["id"] == "shape:1"
    assert merged["x"] == 150
    assert merged["y"] == 100
    assert merged["props"]["text"] == "Initial Text"
    assert merged["props"]["w"] == 200
    assert merged["props"]["h"] == 40


def test_normalize_deletion_event():
    # PostgreSQL pres_annotation_history deletion format (annotationInfo is None)
    raw = {
        "meetingId": "m1",
        "pageId": "p1",
        "annotationId": "shape:del1",
        "annotationInfo": None,
        "updatedAt": 125.0,
    }
    norm = normalize_event(raw)
    assert norm.shape_id == "shape:del1"
    assert norm.is_deleted is True
    assert norm.timestamp_undo == 125.0


def test_normalize_events_sequence():
    events = [
        # 1. Create text shape
        {
            "id": "shape:seq1",
            "type": "text",
            "x": 50,
            "y": 60,
            "timestamp": 10.0,
            "props": {"text": "Kinetic Energy", "w": 100, "h": 30},
        },
        # 2. Delta update: moves position, no text in props
        {
            "id": "shape:seq1",
            "x": 80,
            "y": 90,
            "timestamp": 15.0,
            "props": {"w": 120},
        },
        # 3. Deletion event
        {
            "id": "shape:seq1",
            "timestamp": 25.0,
            "is_deleted": True,
        },
    ]
    normalized = normalize_events(events, lecture_id="m1", slide_id="p1")
    assert len(normalized) == 3

    assert normalized[0].text_content == "Kinetic Energy"
    assert normalized[0].bounds.x == 50.0

    # Delta update retains previous text content and height via state cache
    assert normalized[1].text_content == "Kinetic Energy"
    assert normalized[1].bounds.x == 80.0
    assert normalized[1].bounds.width == 120.0
    assert normalized[1].bounds.height == 30.0

    # Deletion correctly flagged
    assert normalized[2].is_deleted is True
    assert normalized[2].timestamp_undo == 25.0


def test_normalize_malformed_event_resilience():
    # Missing fields, unexpected types
    raw = {"some_random_key": 42}
    norm = normalize_event(raw)
    assert norm.shape_id.startswith("shape:")
    assert norm.shape_type == ShapeType.UNKNOWN
    assert norm.bounds == BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)
    assert norm.is_deleted is False
    assert norm.text_content is None
