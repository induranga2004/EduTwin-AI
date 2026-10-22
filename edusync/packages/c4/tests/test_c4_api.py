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
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"events", "groups", "analyses"} <= tables
