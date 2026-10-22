"""Phase 5 tests: SQLite persistence and the C4 API. All data here is SYNTHETIC / MOCK."""

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT, AnnotationRepository
from edusync_c4.mock_data import MOCK_ANNOTATION_EVENTS, MOCK_LECTURE_ID, MOCK_SLIDE_ID
from edusync_c4.router import get_repository
from edusync_core import CoreSettings, get_settings

L, S = MOCK_LECTURE_ID, MOCK_SLIDE_ID
BASE = "/api/c4"


@pytest.fixture
def repo(tmp_path):
    return AnnotationRepository(tmp_path / "c4" / "annotations.db")


@pytest.fixture
def client(repo):
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def _import(client, events=None, **extra):
    body = {"lecture_id": L, "slide_id": S, "events": events or MOCK_ANNOTATION_EVENTS, **extra}
    return client.post(f"{BASE}/events/import", json=body)


def _txt(shape, ts, text="x", **extra):
    return {"id": shape, "type": "text", "timestamp": ts, "props": {"text": text}, **extra}


def test_existing_endpoints_unchanged(client):
    assert client.get(f"{BASE}/health").json() == {"component": "c4", "status": "ok"}
    resp = client.post(f"{BASE}/reconcile", json={"speech": [], "visual": []})
    assert resp.status_code == 200
    assert resp.json() == {"flags": []}


def test_import_counts_and_idempotence(client, repo):
    first = _import(client).json()
    assert first["imported_count"] == 3
    assert len(first["event_ids"]) == 3
    again = _import(client).json()
    assert again["imported_count"] == 0
    assert again["duplicate_count"] == 3
    events = client.get(f"{BASE}/lectures/{L}/events").json()
    assert len(events) == 3


def test_conflicting_client_event_id_is_reported_not_overwritten(client):
    a = _txt("s", 1, "original", event_id="evt-1")
    b = _txt("s", 1, "different", event_id="evt-1")
    assert _import(client, [a]).json()["imported_count"] == 1
    out = _import(client, [b]).json()
    assert out["imported_count"] == 0
    assert out["conflict_count"] == 1
    assert out["conflict_event_ids"] == ["evt-1"]
    events = client.get(f"{BASE}/lectures/{L}/events", params={"include_raw": True}).json()
    assert events[0]["text_content"] == "original"


def test_malformed_and_unsupported_events_are_stored_and_counted(client):
    events = [
        _txt("ok", 1),
        {"id": "no-time", "type": "text", "props": {"text": "q"}},
        {"id": "odd", "type": "mystery-shape", "timestamp": 2},
        {"id": "ok", "timestamp": 3, "is_deleted": True},
        {"eventName": "ClearWhiteboardEvent", "timestamp": 4},
    ]
    out = _import(client, events).json()
    assert out["imported_count"] == 5
    assert out["unplaced_count"] == 1
    assert out["unsupported_count"] == 1  # only "mystery-shape"; the unplaced one is plain text
    assert out["deleted_count"] == 1
    assert out["clear_count"] == 1
    stored = client.get(f"{BASE}/lectures/{L}/events").json()
    assert len(stored) == 5  # nothing dropped


def test_non_object_event_is_a_validation_error(client):
    resp = client.post(
        f"{BASE}/events/import", json={"lecture_id": L, "slide_id": S, "events": ["nope"]}
    )
    assert resp.status_code == 422


def test_batch_size_limit(client):
    events = [{"id": f"s{i}", "type": "text", "timestamp": 1} for i in range(5001)]
    assert _import(client, events).status_code == 422


def test_data_survives_reopening_the_database(tmp_path):
    path = tmp_path / "x" / "a.db"
    AnnotationRepository(path).import_events(MOCK_ANNOTATION_EVENTS, L, S)
    reopened = AnnotationRepository(path)
    assert reopened.lecture_exists(L)
    timeline = reopened.timeline(L, S)
    assert len(timeline.events) == 3


def test_events_filter_by_slide_and_redaction(client):
    _import(client, [{**_txt("a", 1), "pageId": "p1", "userId": "u-secret"}])
    _import(client, [{**_txt("b", 2), "pageId": "p2"}])
    p1 = client.get(f"{BASE}/lectures/{L}/events", params={"slide_id": "p1"}).json()
    assert [e["shape_id"] for e in p1] == ["a"]
    assert p1[0]["user_id"] == "hidden"
    assert p1[0]["raw_payload"] == {}
    raw = client.get(
        f"{BASE}/lectures/{L}/events", params={"slide_id": "p1", "include_raw": True}
    ).json()
    assert raw[0]["user_id"] == "u-secret"
    assert raw[0]["raw_payload"]["id"] == "a"


def test_unknown_lecture_and_slide_return_404(client):
    assert client.get(f"{BASE}/lectures/nope/events").status_code == 404
    assert client.get(f"{BASE}/lectures/nope/groups").status_code == 404
    assert client.get(f"{BASE}/lectures/nope/catchup").status_code == 404
    _import(client)
    assert client.get(f"{BASE}/lectures/{L}/slides/zzz/snapshot?timestamp=1").status_code == 404
    assert client.post(f"{BASE}/lectures/{L}/slides/zzz/group").status_code == 404


def test_snapshot_handles_delete_readd_and_clear(client):
    events = [
        _txt("a", 1, "v1"),
        {"id": "a", "timestamp": 2, "is_deleted": True},
        _txt("a", 3, "v2"),
        _txt("b", 4, "bb"),
        {"eventName": "ClearWhiteboardEvent", "timestamp": 6},
        _txt("c", 8, "cc"),
    ]
    _import(client, events)

    def at(t):
        r = client.get(f"{BASE}/lectures/{L}/slides/{S}/snapshot", params={"timestamp": t}).json()
        return sorted(a["text_content"] for a in r["annotations"])

    assert at(1.5) == ["v1"]
    assert at(2.5) == []
    assert at(5) == ["bb", "v2"]
    assert at(7) == []
    assert at(9) == ["cc"]
    assert client.get(f"{BASE}/lectures/{L}/slides/{S}/snapshot?timestamp=-1").status_code == 422


def test_delta_across_separate_imports_is_merged(client):
    _import(client, [{"id": "s", "type": "text", "timestamp": 1, "props": {"text": "Force"}}])
    _import(client, [{"id": "s", "timestamp": 5, "x": 40, "props": {"w": 90}}])  # delta only
    snap = client.get(
        f"{BASE}/lectures/{L}/slides/{S}/snapshot", params={"timestamp": 6, "include_raw": True}
    ).json()
    (ann,) = snap["annotations"]
    assert ann["text_content"] == "Force"
    assert ann["bounds"]["x"] == 40.0
    assert ann["bounds"]["width"] == 90.0


def test_group_compute_retrieve_and_invalidation(client):
    _import(client)
    groups = client.post(f"{BASE}/lectures/{L}/slides/{S}/group").json()
    assert len(groups) == 2
    stored = client.get(f"{BASE}/lectures/{L}/groups", params={"slide_id": S}).json()
    assert stored == groups
    loose = client.post(
        f"{BASE}/lectures/{L}/slides/{S}/group",
        json={"max_distance": 5000, "max_gap_seconds": 1000},
    ).json()
    assert len(loose) == 1
    assert len(client.get(f"{BASE}/lectures/{L}/groups").json()) == 1
    _import(client, [_txt("new", 500, "later")])  # new events invalidate cached groups
    assert client.get(f"{BASE}/lectures/{L}/groups").json() == []


def test_analyze_uses_mock_defaults_and_is_labelled(client):
    _import(client)
    resp = client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze")
    assert resp.status_code == 200
    body = resp.json()
    assert body["inputs_are_mock"] == {
        "speech": True,
        "slide_text": True,
        "concepts": True,
        "c2": True,
    }
    assert len(body["analyses"]) == 2
    first = body["analyses"][0]
    assert first["interpretation"]["needs_attention"] is True
    assert "mock-c2" in first["interpretation"]["summary"]


def test_analyze_with_supplied_inputs_and_without_c2(client):
    _import(client)
    body = {
        "speech": [{"id": "x", "start": 118, "end": 126, "text": "Work is force times distance."}],
        "slide_text": "Work and energy.",
        "concepts": [{"id": "work", "label": "Work"}],
        "use_mock_c2": False,
    }
    out = client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze", json=body).json()
    assert out["inputs_are_mock"] == {
        "speech": False,
        "slide_text": False,
        "concepts": False,
        "c2": False,
    }
    first = out["analyses"][0]["interpretation"]
    assert first["needs_attention"] is False
    assert first["comprehension"] == []
    assert "mock-c2" not in first["summary"]


def test_catchup_is_time_ordered_and_filters_by_since(client):
    _import(client)
    client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze")
    everything = client.get(f"{BASE}/lectures/{L}/catchup").json()
    starts = [i["start_seconds"] for i in everything["items"]]
    assert starts == sorted(starts)
    assert len(everything["items"]) == 2
    assert "prerequisite" in everything["note"]
    later = client.get(f"{BASE}/lectures/{L}/catchup", params={"since": 300}).json()
    assert [i["start_seconds"] for i in later["items"]] == [410.0]
    assert client.get(f"{BASE}/lectures/{L}/catchup?since=-5").status_code == 422


def test_catchup_empty_before_analysis(client):
    _import(client)
    assert client.get(f"{BASE}/lectures/{L}/catchup").json()["items"] == []


def test_time_unit_and_lecture_start_offset(client):
    ev = {"id": "s", "type": "text", "updatedAt": 1_700_000_100_000, "props": {"text": "t"}}
    _import(client, [ev], lecture_start_ms=1_700_000_000_000)
    stored = client.get(f"{BASE}/lectures/{L}/events").json()
    assert stored[0]["timestamp_seconds"] == 100.0


def test_database_lives_under_data_dir(tmp_path):
    settings = CoreSettings(data_dir=tmp_path)
    repo = get_repository(settings)
    repo.import_events([_txt("s", 1)], L, S)
    assert (tmp_path / "c4" / "annotations.db").is_file()
    assert get_settings is not None


def test_sqlite_schema_version_and_foreign_keys(repo):
    with sqlite3.connect(repo.db_path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"events", "groups", "analyses", "slides", "speech_segments", "speech_meta"} <= tables


def test_slide_ids_containing_slashes_work_in_paths(client):
    # Real BBB page ids look like "<presentationId>-<timestamp>/<pageNumber>" (found in a live test)
    slide = "2240cd5f-1792662963830/2"
    events = [
        {**_txt("a", 1, "hello"), "pageId": slide},
        {**_txt("b", 2, "world"), "pageId": slide},
    ]
    assert _import(client, events).json()["imported_count"] == 2
    for form in (slide, slide.replace("/", "%2F")):
        base = f"{BASE}/lectures/{L}/slides/{form}"
        snap = client.get(f"{base}/snapshot", params={"timestamp": 5})
        assert snap.status_code == 200
        assert snap.json()["slide_id"] == slide
        assert snap.json()["active_shape_count"] == 2
        assert len(client.post(f"{base}/group").json()) == 1
        assert client.post(f"{base}/analyze").status_code == 200
    assert client.get(f"{BASE}/lectures/{L}/slides/other/2/snapshot?timestamp=1").status_code == 404


# --- Demo Step 1: slides and lecture reset ---

SLIDE = "pres-1/2"  # BBB page ids contain a slash
SVG = '<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"></svg>'


def _put_slide(client, slide=SLIDE, **body):
    return client.put(f"{BASE}/lectures/{L}/slides/{slide}", json=body)


def test_slide_roundtrip_with_slash_id(client):
    resp = _put_slide(
        client,
        number=2,
        width=960,
        height=540,
        scaled_width=1440,
        scaled_height=810,
        text="High Availability",
        svg=SVG,
    )
    assert resp.status_code == 200
    info = resp.json()
    assert info["slide_id"] == SLIDE
    assert info["has_svg"] is True
    assert info["scaled_width"] == 1440
    listing = client.get(f"{BASE}/lectures/{L}/slides").json()
    assert [s["slide_id"] for s in listing] == [SLIDE]
    assert "svg" not in listing[0]  # list never carries the image
    svg = client.get(f"{BASE}/lectures/{L}/slides/{SLIDE}/svg")
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    assert svg.headers["content-security-policy"] == "sandbox"
    assert svg.text == SVG


def test_slide_upsert_keeps_fields_not_resent(client):
    _put_slide(client, number=2, text="first", svg=SVG)
    again = _put_slide(client, text="second").json()
    assert again["text"] == "second"
    assert again["number"] == 2
    assert again["has_svg"] is True


def test_slides_are_ordered_by_number(client):
    for n in (3, 1, 2):
        _put_slide(client, slide=f"p/{n}", number=n)
    assert [s["number"] for s in client.get(f"{BASE}/lectures/{L}/slides").json()] == [1, 2, 3]


@pytest.mark.parametrize(
    "bad",
    [
        {"svg": "<html></html>"},
        {"svg": '<!DOCTYPE svg [<!ENTITY a "b">]><svg></svg>'},
        {"svg": "<svg>" + "x" * 5_000_001},
        {"width": -1},
        {"number": -3},
    ],
)
def test_invalid_slide_payloads_are_rejected(client, bad):
    assert _put_slide(client, **bad).status_code == 422


def test_missing_slide_image_is_404(client):
    assert client.get(f"{BASE}/lectures/{L}/slides/none/svg").status_code == 404
    _put_slide(client, number=1)  # no svg stored
    assert client.get(f"{BASE}/lectures/{L}/slides/{SLIDE}/svg").status_code == 404
    assert client.get(f"{BASE}/lectures/unknown/slides").json() == []


def test_delete_lecture_removes_everything_and_only_that_lecture(client):
    _import(client)
    client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze")
    _put_slide(client, svg=SVG)
    other = {"lecture_id": "keep", "slide_id": S, "events": [_txt("k", 1)]}
    client.post(f"{BASE}/events/import", json=other)
    counts = client.delete(f"{BASE}/lectures/{L}").json()
    assert counts == {
        "groups": 2,
        "slides": 1,
        "events": 3,
        "slide_words": 0,
        "speech_segments": 0,
        "speech_meta": 0,
    }
    assert client.get(f"{BASE}/lectures/{L}/events").status_code == 404
    assert client.get(f"{BASE}/lectures/{L}/slides").json() == []
    assert client.get(f"{BASE}/lectures/{L}/catchup").status_code == 404
    assert len(client.get(f"{BASE}/lectures/keep/events").json()) == 1
    assert client.delete(f"{BASE}/lectures/nothing").json() == dict.fromkeys(
        ("groups", "slides", "slide_words", "events", "speech_segments", "speech_meta"), 0
    )
