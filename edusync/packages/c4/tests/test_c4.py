import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT
from edusync_c4.annotations import parse_annotation_text
from edusync_c4.evidence import merge_evidence, reconcile
from edusync_c4.grouping import (
    SpatioTemporalClusterer,
    box_distance,
    group_timeline,
    time_gap,
)
from edusync_c4.normalizer import (
    apply_delta,
    normalize_event,
    normalize_events,
)
from edusync_c4.schemas import (
    BoundingBox,
    GroupingConfig,
    ShapeType,
)
from edusync_c4.timeline import (
    build_timeline,
    build_timelines,
    reconstruct_snapshot,
    replay_timeline,
)
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
        "updatedAt": 125000.0,  # BBB updatedAt is epoch milliseconds
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


# --- Phase 2: Temporal Reconstruction & Interval Tests ---


def test_build_timeline_interval_chaining():
    events = [
        {
            "id": "shape:dyn1",
            "type": "text",
            "x": 10.0,
            "y": 20.0,
            "timestamp": 10.0,
            "props": {"text": "v1"},
        },
        {
            "id": "shape:dyn1",
            "type": "text",
            "x": 15.0,
            "y": 25.0,
            "timestamp": 20.0,
            "props": {"text": "v2"},
        },
        {
            "id": "shape:dyn1",
            "type": "text",
            "x": 20.0,
            "y": 30.0,
            "timestamp": 35.0,
            "props": {"text": "v3"},
        },
    ]
    timeline = build_timeline(events, lecture_id="lec1", slide_id="slide1")
    assert timeline.lecture_id == "lec1"
    assert timeline.slide_id == "slide1"
    assert len(timeline.events) == 3

    v1, v2, v3 = timeline.events
    assert v1.text_content == "v1"
    assert v1.timestamp_seconds == 10.0
    assert v1.timestamp_out == 20.0
    assert v1.timestamp_undo is None

    assert v2.text_content == "v2"
    assert v2.timestamp_seconds == 20.0
    assert v2.timestamp_out == 35.0
    assert v2.timestamp_undo is None

    assert v3.text_content == "v3"
    assert v3.timestamp_seconds == 35.0
    assert v3.timestamp_out is None
    assert v3.timestamp_undo is None


def test_build_timeline_deletion_undo():
    events = [
        {
            "id": "shape:del_test",
            "type": "geo",
            "x": 0.0,
            "y": 0.0,
            "timestamp": 5.0,
            "props": {"w": 100, "h": 50},
        },
        {
            "id": "shape:del_test",
            "x": 10.0,
            "y": 10.0,
            "timestamp": 15.0,
            "props": {"w": 120},
        },
        {
            "id": "shape:del_test",
            "timestamp": 28.0,
            "is_deleted": True,
        },
    ]
    timeline = build_timeline(events, lecture_id="lec1", slide_id="slide1")
    v1 = timeline.events[0]
    v2 = timeline.events[1]
    v_del = timeline.events[2]

    assert v1.timestamp_undo == 28.0
    assert v2.timestamp_undo == 28.0
    assert v_del.is_deleted is True
    assert v_del.timestamp_undo == 28.0


def test_reconstruct_snapshot_point_in_time():
    events = [
        {
            "id": "shape:target",
            "type": "text",
            "x": 10.0,
            "y": 10.0,
            "timestamp": 10.0,
            "props": {"text": "Version A"},
        },
        {
            "id": "shape:target",
            "type": "text",
            "x": 20.0,
            "y": 20.0,
            "timestamp": 20.0,
            "props": {"text": "Version B"},
        },
        {
            "id": "shape:target",
            "timestamp": 30.0,
            "is_deleted": True,
        },
    ]
    timeline = build_timeline(events, lecture_id="m1", slide_id="p1")

    # Before creation (t = 5.0): nothing visible
    s5 = reconstruct_snapshot(timeline, 5.0)
    assert s5.active_shape_count == 0
    assert s5.annotations == []

    # Exactly at creation (t = 10.0): Version A visible
    s10 = reconstruct_snapshot(timeline, 10.0)
    assert s10.active_shape_count == 1
    assert s10.annotations[0].text_content == "Version A"

    # Mid Version A (t = 15.0): Version A visible
    s15 = reconstruct_snapshot(timeline, 15.0)
    assert s15.active_shape_count == 1
    assert s15.annotations[0].text_content == "Version A"

    # Exactly at update (t = 20.0): Version B visible
    s20 = reconstruct_snapshot(timeline, 20.0)
    assert s20.active_shape_count == 1
    assert s20.annotations[0].text_content == "Version B"

    # Mid Version B (t = 25.0): Version B visible
    s25 = reconstruct_snapshot(timeline, 25.0)
    assert s25.active_shape_count == 1
    assert s25.annotations[0].text_content == "Version B"

    # Exactly at deletion (t = 30.0): retracted, nothing visible
    s30 = reconstruct_snapshot(timeline, 30.0)
    assert s30.active_shape_count == 0

    # Long after deletion (t = 100.0): nothing visible
    s100 = reconstruct_snapshot(timeline, 100.0)
    assert s100.active_shape_count == 0


def test_timeline_multiple_shapes_interleaving():
    events = [
        # Shape A: created t=5, updated t=15
        {"id": "shape:A", "type": "text", "timestamp": 5.0, "props": {"text": "A1"}},
        {"id": "shape:A", "type": "text", "timestamp": 15.0, "props": {"text": "A2"}},
        # Shape B: created t=10, deleted t=25
        {"id": "shape:B", "type": "note", "timestamp": 10.0, "props": {"text": "B1"}},
        {"id": "shape:B", "timestamp": 25.0, "is_deleted": True},
        # Shape C: created t=20
        {"id": "shape:C", "type": "arrow", "timestamp": 20.0, "props": {}},
    ]
    timeline = build_timeline(events, lecture_id="lec", slide_id="slide")

    # t = 7: Shape A only (A1)
    active_7 = [a.shape_id for a in timeline.get_active_annotations(7.0)]
    assert active_7 == ["shape:A"]
    assert timeline.get_active_annotations(7.0)[0].text_content == "A1"

    # t = 12: Shape A (A1) and Shape B (B1)
    active_12 = sorted(a.shape_id for a in timeline.get_active_annotations(12.0))
    assert active_12 == ["shape:A", "shape:B"]

    # t = 18: Shape A (A2) and Shape B (B1)
    s18 = reconstruct_snapshot(timeline, 18.0)
    assert s18.active_shape_count == 2
    a_shape = next(a for a in s18.annotations if a.shape_id == "shape:A")
    assert a_shape.text_content == "A2"

    # t = 22: Shape A (A2), Shape B (B1), Shape C
    active_22 = sorted(a.shape_id for a in timeline.get_active_annotations(22.0))
    assert active_22 == ["shape:A", "shape:B", "shape:C"]

    # t = 28: Shape B is deleted; Shape A (A2) and Shape C active
    active_28 = sorted(a.shape_id for a in timeline.get_active_annotations(28.0))
    assert active_28 == ["shape:A", "shape:C"]


def test_timeline_keyframes_and_replay():
    events = [
        {"id": "shape:X", "type": "text", "timestamp": 10.0, "props": {"text": "X"}},
        {"id": "shape:Y", "type": "text", "timestamp": 20.0, "props": {"text": "Y"}},
        {"id": "shape:X", "timestamp": 30.0, "is_deleted": True},
    ]
    timeline = build_timeline(events)
    keyframes = timeline.get_keyframes()
    assert keyframes == [10.0, 20.0, 30.0]

    snapshots = replay_timeline(timeline, slide_text="Background Slide Text")
    assert len(snapshots) == 3
    assert snapshots[0].timestamp_seconds == 10.0
    assert snapshots[0].active_shape_count == 1
    assert snapshots[0].slide_text == "Background Slide Text"

    assert snapshots[1].timestamp_seconds == 20.0
    assert snapshots[1].active_shape_count == 2

    assert snapshots[2].timestamp_seconds == 30.0
    assert snapshots[2].active_shape_count == 1
    assert snapshots[2].annotations[0].shape_id == "shape:Y"


def test_build_timeline_empty():
    timeline = build_timeline([], lecture_id="none", slide_id="none")
    assert timeline.events == []
    assert timeline.get_keyframes() == []
    assert timeline.get_active_annotations(100.0) == []
    assert reconstruct_snapshot(timeline, 50.0).active_shape_count == 0
    assert replay_timeline(timeline) == []


# --- Phase 2 audit regressions (all fixtures are SYNTHETIC, shaped like BBB rows) ---


def _txt(shape, ts, text="x", **extra):
    return {"id": shape, "type": "text", "timestamp": ts, "props": {"text": text}, **extra}


def _del(shape, ts):
    return {"id": shape, "timestamp": ts, "is_deleted": True}


def _texts(timeline, t):
    return [(a.shape_id, a.text_content) for a in timeline.get_active_annotations(t)]


def test_reappear_after_delete_then_deleted_again():
    # create -> delete -> re-add (undo/redo) -> delete: must be gone after the 2nd delete
    tl = build_timeline([_txt("s", 1, "a"), _del("s", 2), _txt("s", 3, "b"), _del("s", 4)])
    assert _texts(tl, 1.5) == [("s", "a")]
    assert _texts(tl, 2.5) == []
    assert _texts(tl, 3.5) == [("s", "b")]
    assert _texts(tl, 10) == []


def test_whiteboard_clear_retracts_all_shapes_and_is_deterministic():
    clear = {"eventName": "ClearWhiteboardEvent", "timestamp": 5, "fullClear": True}
    events = [_txt("a", 1), _txt("b", 2), clear, _txt("c", 7)]
    tl = build_timeline(events)
    assert sorted(s for s, _ in _texts(tl, 4)) == ["a", "b"]
    assert _texts(tl, 6) == []
    assert [s for s, _ in _texts(tl, 8)] == ["c"]
    clears = [e for e in tl.events if e.is_clear]
    assert len(clears) == 1
    assert clears[0].shape_id == "clear:default"
    # same input -> same ids (idempotent import)
    assert [e.event_id for e in build_timeline(events).events] == [e.event_id for e in tl.events]


def test_partial_clear_only_removes_clearing_users_shapes():
    clear = {
        "eventName": "ClearWhiteboardEvent",
        "timestamp": 5,
        "fullClear": False,
        "userId": "u1",
    }
    tl = build_timeline(
        [_txt("a", 1, userId="u1"), _txt("b", 2, userId="u2"), clear],
    )
    assert [s for s, _ in _texts(tl, 6)] == ["b"]


def test_bbb_history_row_deletion_and_millisecond_timestamps():
    rows = [
        {
            "annotationId": "s",
            "annotationInfo": {"id": "s", "type": "text", "props": {"text": "a"}},
            "updatedAt": 1000,
        },
        {"annotationId": "s", "annotationInfo": None, "updatedAt": 2000},
    ]
    tl = build_timeline(rows, lecture_start_ms=1000)
    assert [e.timestamp_seconds for e in tl.events] == [0.0, 1.0]
    assert _texts(tl, 0.5) == [("s", "a")]
    assert _texts(tl, 1.5) == []


def test_timestamp_units_are_explicit_not_guessed():
    # 12000 elapsed seconds (3h20m) must stay 12000 s, small ms values stay ms
    assert normalize_event(_txt("s", 12000)).timestamp_seconds == 12000.0
    assert normalize_event(_txt("s", 12000), time_unit="ms").timestamp_seconds == 12.0
    assert normalize_event({"id": "s", "type": "text", "updatedAt": 1500}).timestamp_seconds == 1.5


def test_missing_timestamp_is_flagged_not_zeroed():
    norm = normalize_event({"id": "s", "type": "text", "props": {"text": "q"}})
    assert norm.timestamp_valid is False
    tl = build_timeline([{"id": "s", "type": "text", "props": {"text": "q"}}])
    assert tl.events == []
    assert len(tl.unplaced) == 1
    assert tl.get_active_annotations(0.0) == []


def test_zero_duration_versions_are_kept():
    tl = build_timeline([_txt("s", 10, "a"), _txt("s", 10, "b")])
    assert len(tl.events) == 2
    assert _texts(tl, 10) == [("s", "b")]


def test_build_timeline_does_not_mutate_inputs():
    norm = normalize_events([_txt("s", 1), _txt("s", 5)])
    build_timeline(norm)
    assert norm[0].timestamp_out is None
    tl = build_timeline(norm)
    tl.get_active_annotations(2.0)[0].text_content = "changed"
    assert tl.events[0].text_content == "x"


def test_mixed_raw_and_normalized_input():
    pre = normalize_event(_txt("a", 1, "pre"))
    tl = build_timeline([pre, _txt("b", 2, "raw")])
    assert sorted(s for s, _ in _texts(tl, 3)) == ["a", "b"]


def test_multiple_slides_are_split_not_merged():
    events = [_txt("s1", 1, pageId="p1"), _txt("s2", 2, pageId="p2")]
    with pytest.raises(ValueError, match="span 2 slides"):
        build_timeline(events)
    timelines = build_timelines(events)
    assert {k[1] for k in timelines} == {"p1", "p2"}
    assert [e.shape_id for e in timelines[("default", "p1")].events] == ["s1"]


def test_unsorted_delta_input_merges_chronologically():
    events = [
        {"id": "s", "type": "text", "timestamp": 20, "props": {"text": "new"}},
        {"id": "s", "type": "text", "timestamp": 10, "props": {"text": "old", "w": 9}},
    ]
    tl = build_timeline(events)
    old, new = tl.events
    assert old.text_content == "old"
    assert old.bounds.width == 9.0
    assert new.text_content == "new"
    assert new.bounds.width == 9.0  # later delta inherits earlier props
    assert old.timestamp_out == 20


def test_same_millisecond_update_and_delete_have_distinct_event_ids():
    a = normalize_event(_txt("s", 1))
    b = normalize_event(_del("s", 1))
    assert a.event_id != b.event_id


# --- Phase 3: candidate grouping (SYNTHETIC fixtures) ---


def _geo(shape, ts, x, y, w=40, h=20, **extra):
    return {
        "id": shape,
        "type": "geo",
        "x": x,
        "y": y,
        "timestamp": ts,
        "props": {"w": w, "h": h},
        **extra,
    }


def _groups(events, **cfg):
    tl = build_timeline(events, lecture_id="lec", slide_id="sl")
    return group_timeline(tl, GroupingConfig(**cfg) if cfg else None)


def test_box_distance_and_time_gap():
    a = BoundingBox(x=0, y=0, width=10, height=10)
    assert box_distance(a, BoundingBox(x=5, y=5, width=10, height=10)) == 0.0
    assert box_distance(a, BoundingBox(x=13, y=14, width=1, height=1)) == 5.0
    assert time_gap((0, 5), (3, 9)) == 0.0
    assert time_gap((0, 5), (8, 9)) == 3.0


def test_separate_regions_on_same_slide_form_separate_groups():
    events = [
        _geo("a", 1, 0, 0),
        _geo("b", 2, 60, 0),  # 20 units from a -> same group
        _geo("c", 3, 800, 600),  # far away -> own group
    ]
    groups = _groups(events)
    assert [g.shape_ids for g in groups] == [["a", "b"], ["c"]]
    assert groups[0].member_count == 2
    assert groups[0].bounds == BoundingBox(x=0, y=0, width=100, height=20)


def test_same_place_but_far_apart_in_time_is_split():
    events = [_geo("a", 1, 0, 0), _geo("b", 100, 10, 0)]
    assert len(_groups(events)) == 2
    assert len(_groups(events, max_gap_seconds=200)) == 1


def test_thresholds_are_configurable():
    events = [_geo("a", 1, 0, 0), _geo("b", 2, 100, 0)]  # 60 units apart
    assert len(_groups(events)) == 2
    assert len(_groups(events, max_distance=60)) == 1


def test_group_preserves_event_ids_and_native_text():
    events = [
        _txt("t1", 1, "Force", x=0, y=0, props={"text": "Force", "w": 40, "h": 20}),
        _txt("t1", 2, "Force = ma", x=0, y=0),
    ]
    tl = build_timeline(events, slide_id="sl")
    (g,) = group_timeline(tl)
    assert sorted(g.event_ids) == sorted(e.event_id for e in tl.events)
    assert g.native_text == "Force = ma"  # final text only, not every edit
    assert g.primary_type == ShapeType.TEXT


def test_retracted_shapes_kept_by_default_and_optionally_excluded():
    events = [_geo("a", 1, 0, 0), _geo("b", 2, 10, 0), _del("b", 5)]
    (g,) = _groups(events)
    assert g.retracted_shape_ids == ["b"]
    assert g.shape_ids == ["a", "b"]
    (g2,) = _groups(events, include_retracted=False)
    assert g2.shape_ids == ["a"]


def test_clear_and_delete_records_are_not_members():
    events = [_geo("a", 1, 0, 0), {"eventName": "ClearWhiteboardEvent", "timestamp": 4}]
    (g,) = _groups(events)
    assert g.shape_ids == ["a"]


def test_grouping_is_deterministic_and_input_order_independent():
    events = [_geo("a", 1, 0, 0), _geo("b", 2, 30, 0), _geo("c", 3, 700, 700)]
    g1 = _groups(events)
    g2 = _groups(list(reversed(events)))
    assert [g.model_dump() for g in g1] == [g.model_dump() for g in g2]


def test_empty_timeline_and_custom_clusterer_protocol():
    tl = build_timeline([], lecture_id="l", slide_id="s")
    assert group_timeline(tl) == []

    class OneBlob:
        def group(self, timeline, config=None):
            return SpatioTemporalClusterer().group(timeline, GroupingConfig(max_distance=1e9))

    tl2 = build_timeline([_geo("a", 1, 0, 0), _geo("b", 2, 900, 900)])
    assert len(group_timeline(tl2, clusterer=OneBlob())) == 1


def test_single_linkage_chaining_limitation_is_real():
    # a~b and b~c but a and c are 120 apart: single linkage still merges all three
    events = [_geo("a", 1, 0, 0), _geo("b", 2, 60, 0), _geo("c", 3, 120, 0)]
    assert len(_groups(events)) == 1


def test_invalid_config_rejected():
    with pytest.raises(ValueError, match="greater than or equal"):
        GroupingConfig(max_distance=-1)


# --- Phase 4: context, mock C2, recognizer placeholder, interpretation (MOCK data only) ---

import ast  # noqa: E402
from pathlib import Path  # noqa: E402

from edusync_c4 import (  # noqa: E402
    ContextConfig,
    DeterministicDifficultyPredictor,
    ExactPhraseMatcher,
    HeuristicRecognizer,
    SlideText,
    SpeechSegment,
    StaticComprehensionEvaluator,
    TemplateContextualInterpreter,
    analyze_groups,
    build_group_context,
    members_of,
)
from edusync_c4.mock_data import (  # noqa: E402
    MOCK_ANNOTATION_EVENTS,
    MOCK_COMPREHENSION,
    MOCK_CONCEPTS,
    MOCK_LECTURE_ID,
    MOCK_SLIDE_ID,
    MOCK_SLIDE_TEXT,
    MOCK_TRANSCRIPT,
)


def _mock_groups():
    tl = build_timeline(MOCK_ANNOTATION_EVENTS, lecture_id=MOCK_LECTURE_ID, slide_id=MOCK_SLIDE_ID)
    return tl, group_timeline(tl)


def _analyze(**kw):
    tl, groups = _mock_groups()
    return analyze_groups(tl, groups, MOCK_TRANSCRIPT, MOCK_SLIDE_TEXT, MOCK_CONCEPTS, **kw)


def test_exact_phrase_matcher_whole_word_case_insensitive():
    m = ExactPhraseMatcher()
    hits = m.match("the KINETIC ENERGY of a body; workflow", MOCK_CONCEPTS)
    assert [h.concept.id for h in hits] == ["kinetic-energy"]  # 'workflow' must not match 'Work'
    assert hits[0].matched_text == "KINETIC ENERGY"
    assert m.match("anything", []) == []


def test_speech_window_inclusion_and_relations():
    _, groups = _mock_groups()
    g = groups[0]  # drawn 120s..125s
    ctx = build_group_context(g, MOCK_TRANSCRIPT, MOCK_SLIDE_TEXT, MOCK_CONCEPTS)
    assert [(s.segment_id, s.relation) for s in ctx.speech] == [(1, "before"), (2, "during")]
    narrow = build_group_context(
        g, MOCK_TRANSCRIPT, MOCK_SLIDE_TEXT, MOCK_CONCEPTS, ContextConfig(speech_pad_seconds=0)
    )
    assert [s.segment_id for s in narrow.speech] == [2]


def test_speech_offset_moves_transcript_onto_annotation_clock():
    _, groups = _mock_groups()
    cfg = ContextConfig(speech_pad_seconds=0, speech_offset_seconds=300)
    ctx = build_group_context(groups[0], MOCK_TRANSCRIPT, None, MOCK_CONCEPTS, cfg)
    assert ctx.speech == []  # shifted +300s, nothing left near 120s..125s


def test_slide_text_excerpt_truncation_and_wrong_slide():
    _, groups = _mock_groups()
    short = ContextConfig(max_slide_excerpt_chars=20)
    ctx = build_group_context(groups[0], [], MOCK_SLIDE_TEXT, MOCK_CONCEPTS, short)
    assert ctx.slide_text_truncated is True
    assert len(ctx.slide_text_excerpt) <= 20
    other = SlideText(slide_id="other", text="nope")
    ctx2 = build_group_context(groups[0], [], other, MOCK_CONCEPTS)
    assert ctx2.slide_text_excerpt is None
    assert ctx2.slide_evidence_id is None


def test_concept_candidates_come_from_annotation_and_speech_not_whole_slide():
    _, groups = _mock_groups()
    ctx = build_group_context(groups[1], MOCK_TRANSCRIPT, MOCK_SLIDE_TEXT, MOCK_CONCEPTS)
    # group 2 ("Unrelated aside", speech 'different topic') must not inherit slide concepts
    assert ctx.concept_candidates == []
    ctx1 = build_group_context(groups[0], MOCK_TRANSCRIPT, MOCK_SLIDE_TEXT, MOCK_CONCEPTS)
    assert {c.source for c in ctx1.concept_candidates} == {"annotation", "speech"}


def test_mock_c2_is_deterministic_and_tagged():
    p = DeterministicDifficultyPredictor()
    assert p.predict("Kinetic energy") == p.predict("Kinetic energy")
    assert p.predict("").score == 0.0
    assert p.predict("antidisestablishmentarianism").score == 1.0
    assert p.predict("a cat sat").source == "mock-c2"
    ev = StaticComprehensionEvaluator(MOCK_COMPREHENSION)
    assert ev.evaluate("kinetic-energy").struggling is True
    assert ev.evaluate("momentum").struggling is False
    assert ev.evaluate("unknown") is None
    assert StaticComprehensionEvaluator({"x": (1, 2)}).evaluate("x") is None  # too few responses
    assert ev.evaluate("momentum").source == "mock-c2"


def test_recognizer_labels_freehand_as_placeholder():
    tl, groups = _mock_groups()
    rec = HeuristicRecognizer().recognize(groups[0], members_of(groups[0], tl))
    assert rec.kind == "mixed"
    assert rec.freehand_stroke_count == 1
    assert rec.text == "Kinetic energy"
    assert rec.is_placeholder is False  # it has real native text
    assert any("not recognised" in n for n in rec.notes)

    tl2 = build_timeline(
        [{"id": "d", "type": "draw", "timestamp": 1, "props": {"segments": [{"points": [[0, 0]]}]}}]
    )
    (g,) = group_timeline(tl2)
    only = HeuristicRecognizer().recognize(g, members_of(g, tl2))
    assert only.kind == "unrecognized_freehand"
    assert only.text is None
    assert only.is_placeholder is True


def test_pipeline_separates_evidence_from_interpretation_and_cites_it():
    results = _analyze(
        interpreter=TemplateContextualInterpreter(
            DeterministicDifficultyPredictor(), StaticComprehensionEvaluator(MOCK_COMPREHENSION)
        )
    )
    assert len(results) == 2
    first = results[0]
    assert type(first.context) is not type(first.interpretation)
    assert first.interpretation.method == "template-v1"
    assert first.interpretation.confidence <= 0.5
    known = set(first.context.evidence_ids())
    assert set(first.interpretation.cited_evidence_ids) <= known
    assert first.interpretation.category == "text_with_freehand"
    assert first.interpretation.needs_attention is True  # mock quiz: kinetic-energy struggling
    assert "not recognised" in first.interpretation.summary
    assert "mock-c2" in first.interpretation.summary
    assert "Rule-based template" in first.interpretation.limitations[0]


def test_pipeline_without_c2_or_speech_or_slide():
    tl, groups = _mock_groups()
    results = analyze_groups(tl, groups, [], None, [])
    for r in results:
        assert r.interpretation.difficulty is None
        assert r.interpretation.comprehension == []
        assert r.interpretation.needs_attention is False
        assert "No speech was found" in r.interpretation.summary
    assert analyze_groups(tl, [], [], None, []) == []


def test_pipeline_is_deterministic():
    a = [r.model_dump() for r in _analyze()]
    b = [r.model_dump() for r in _analyze()]
    assert a == b


def test_speech_segment_validation():
    with pytest.raises(ValueError, match="greater than or equal"):
        SpeechSegment(id=1, start=-1, end=2, text="x")


def test_c4_does_not_import_other_components():
    pkg = Path(__import__("edusync_c4").__file__).parent
    banned = {"edusync_c1", "edusync_c2", "edusync_c3"}
    for path in pkg.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            assert not {m.split(".")[0] for m in mods} & banned, path.name


# --- Grouping fixes from the live BBB test (SYNTHETIC copy of the real sequence) ---


def _note(shape, ts, text, x=621, y=322, **extra):
    return {
        "id": shape,
        "type": "note",
        "x": x,
        "y": y,
        "timestamp": ts,
        "props": {"text": text},
        **extra,
    }


def _stroke(shape, ts, x, y):
    pts = [[0, 0, 0.5], [40, 10, 0.5], [80, 0, 0.5]]
    return {
        "id": shape,
        "type": "draw",
        "x": x,
        "y": y,
        "timestamp": ts,
        "props": {"segments": [{"points": pts}]},
    }


def test_group_text_is_final_text_not_every_keystroke():
    steps = ["a", "d", "do", "don", "dont", "dont k", "dont know"]
    events = [_note("n", 10 + i, t) for i, t in enumerate(steps)]
    (g,) = _groups(events)
    assert g.native_text == "dont know"


def test_undo_restores_do_not_merge_unrelated_shapes():
    events = [
        {
            "id": "arrow",
            "type": "arrow",
            "timestamp": 0,
            "props": {"start": {"x": 0, "y": 0}, "end": {"x": 24, "y": 44}},
            "x": 727,
            "y": 358,
        },
        _note("n", 4.68, "x"),
        _note("n", 29.4, "dont know", x=730, y=405),
        _geo("g", 838.85, 348, 531, w=261, h=134),
        _del("g", 862.6),
        _del("n", 862.6),
        _note("n", 867.37, "dont know", x=730, y=405),  # undo of the note's delete
        _geo("g", 869.98, 348, 531, w=261, h=134),  # undo of the geo's delete
        _del("g", 871.06),
    ]
    groups = _groups(events)
    by_members = sorted(g.shape_ids for g in groups)
    assert by_members == [["arrow", "n"], ["g"]]


def test_restore_after_clear_is_not_an_authored_edit():
    clear = {"eventName": "ClearWhiteboardEvent", "timestamp": 50}
    events = [
        _geo("a", 1, 0, 0),
        _geo("b", 100, 20, 0),
        clear,
        _geo("a", 102, 0, 0),  # undo of the clear brings 'a' back at 102 s
    ]
    # 'a' was drawn at 1 s and 'b' at 100 s: far apart. The restore at 102 s must not link them.
    assert len(_groups(events)) == 2


def test_real_edit_after_the_gap_still_links():
    events = [_geo("a", 1, 0, 0), _geo("b", 100, 20, 0), _geo("a", 101, 0, 0)]
    # no deletion in between: 'a' was really edited at 101 s, next to 'b' at 100 s
    assert len(_groups(events)) == 1


def test_partial_clear_only_affects_the_clearing_users_shapes():
    def run(clearing_user):
        clear = {
            "eventName": "ClearWhiteboardEvent",
            "timestamp": 50,
            "fullClear": False,
            "userId": clearing_user,
        }
        return _groups(
            [
                _geo("a", 1, 0, 0, userId="u1"),
                _geo("b", 100, 20, 0, userId="u2"),
                clear,
                _geo("a", 102, 0, 0, userId="u1"),
            ]
        )

    # u1 cleared their own shape 'a': its return at 102 s is a restore, so 'a' stays far from 'b'
    assert len(run("u1")) == 2
    # u2 cleared, which never touched u1's 'a': the edit at 102 s is genuine and links to 'b'
    assert len(run("u2")) == 1
