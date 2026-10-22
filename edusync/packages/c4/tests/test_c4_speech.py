"""Demo Step 4 tests: stored speech, clock alignment (manual offset / sync word) and the report.

All data is SYNTHETIC.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT, AnnotationRepository
from edusync_c4.report import MAX_QUOTES, said_at
from edusync_c4.router import get_repository
from edusync_c4.schemas import SpeechSegment

L, S = "lec", "pres/1"
BASE = "/api/c4"


@pytest.fixture
def client(tmp_path):
    repo = AnnotationRepository(tmp_path / "c4" / "annotations.db")
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def text_event(shape, ts, text, x=100, y=100, page=S):
    props = {"text": text, "w": 200, "h": 30}
    return {
        "id": shape,
        "type": "text",
        "x": x,
        "y": y,
        "timestamp": ts,
        "props": props,
        "pageId": page,
    }


def load(client, events):
    body = {"lecture_id": L, "slide_id": S, "events": events}
    assert client.post(f"{BASE}/events/import", json=body).status_code == 200


def seg(i, start, end, text):
    return {"id": i, "start": start, "end": end, "text": text}


def put(client, segments, **extra):
    return client.put(f"{BASE}/lectures/{L}/speech", json={"segments": segments, **extra})


def test_manual_offset_moves_speech_onto_the_meeting_clock(client):
    resp = put(client, [seg(0, 1, 4, "hello"), seg(1, 10, 12, "world")], offset_seconds=100)
    assert resp.status_code == 200
    assert resp.json()["method"] == "manual"
    got = client.get(f"{BASE}/lectures/{L}/speech").json()
    assert [(s["start"], s["end"]) for s in got["segments"]] == [(101.0, 104.0), (110.0, 112.0)]
    assert got["offset_seconds"] == 100.0
    assert got["segment_count"] == 2


def test_no_alignment_given_means_offset_zero(client):
    info = put(client, [seg(0, 5, 6, "x")]).json()
    assert (info["offset_seconds"], info["method"]) == (0.0, "none")


def test_sync_word_aligns_the_clocks_and_hides_the_sync_mark(client):
    load(
        client,
        [
            text_event("dot", 100.0, "."),  # the mark drawn at the moment "sync" was said
            text_event("real", 130.0, "Kinetic energy"),
        ],
    )
    segments = [seg(0, 2, 5, "okay let us start"), seg(1, 12, 13, "sync"), seg(2, 40, 44, "energy")]
    info = put(client, segments, sync_word="sync").json()
    assert info["method"] == "sync_word"
    assert info["offset_seconds"] == 88.0  # 100 (drawn) - 12 (spoken)
    assert info["sync_shape_id"] == "dot"
    got = client.get(f"{BASE}/lectures/{L}/speech").json()
    assert got["segments"][1]["start"] == 100.0
    shapes = [e["shape_id"] for e in client.get(f"{BASE}/lectures/{L}/events").json()]
    assert shapes == ["real"]  # the cue mark is not lecture content
    report = client.get(f"{BASE}/lectures/{L}/report").text
    assert "Kinetic energy" in report
    assert report.count("<g transform") == 1  # only the real item is drawn, not the cue mark


@pytest.mark.parametrize("spoken", ["Sync", "and now sink", "SYNCH!", "sync."])
def test_sync_word_tolerates_common_mishearings(client, spoken):
    load(client, [text_event("dot", 50.0, "."), text_event("x", 60.0, "x")])
    resp = put(client, [seg(0, 5, 6, spoken)], sync_word="sync")
    assert resp.status_code == 200
    assert resp.json()["offset_seconds"] == 45.0


def test_sync_word_must_be_a_whole_word(client):
    load(client, [text_event("dot", 50.0, ".")])
    resp = put(client, [seg(0, 5, 6, "a synchronous system")], sync_word="sync")
    assert resp.status_code == 422
    assert "not found" in resp.json()["detail"]


def test_sync_needs_annotations_and_a_known_lecture(client):
    assert put(client, [seg(0, 1, 2, "sync")], sync_word="sync").status_code == 422


def test_other_sync_words_match_exactly(client):
    load(client, [text_event("dot", 50.0, "."), text_event("x", 60.0, "x")])
    assert put(client, [seg(0, 5, 6, "go banana go")], sync_word="banana").status_code == 200
    assert put(client, [seg(0, 5, 6, "go bananas")], sync_word="banana").status_code == 422


def test_upload_replaces_previous_speech_and_validates(client):
    put(client, [seg(0, 1, 2, "old one"), seg(1, 3, 4, "old two")])
    put(client, [seg(0, 1, 2, "new")])
    got = client.get(f"{BASE}/lectures/{L}/speech").json()
    assert [s["text"] for s in got["segments"]] == ["new"]
    assert put(client, [seg(0, -1, 2, "bad")]).status_code == 422
    assert put(client, [seg(i, i, i + 1, "x") for i in range(5001)]).status_code == 422
    assert put(client, [seg(0, 1, 2, "x")], sync_word="x").status_code == 422  # too short a word


def test_unknown_lecture_speech_is_404(client):
    assert client.get(f"{BASE}/lectures/none/speech").status_code == 404


def test_lecture_list_counts_and_orders_newest_first(client):
    load(client, [text_event("a", 5.0, "one"), text_event("b", 9.0, "two")])
    other = {
        "lecture_id": "newer",
        "slide_id": "s",
        "events": [text_event("c", 1.0, "x", page="s")],
    }
    client.post(f"{BASE}/events/import", json=other)
    put(client, [seg(0, 1, 2, "hi")])
    lectures = client.get(f"{BASE}/lectures").json()
    assert lectures[0]["lecture_id"] == "newer"
    mine = next(x for x in lectures if x["lecture_id"] == L)
    assert (mine["event_count"], mine["slide_count"]) == (2, 1)
    assert (mine["first_seconds"], mine["last_seconds"]) == (5.0, 9.0)
    assert mine["has_speech"] is True
    assert next(x for x in lectures if x["lecture_id"] == "newer")["has_speech"] is False


def test_deleting_a_lecture_removes_its_speech(client):
    load(client, [text_event("a", 5.0, "one")])
    put(client, [seg(0, 1, 2, "hi")])
    counts = client.delete(f"{BASE}/lectures/{L}").json()
    assert counts["speech_segments"] == 1
    assert counts["speech_meta"] == 1
    assert client.get(f"{BASE}/lectures/{L}/speech").status_code == 404


def test_new_speech_invalidates_cached_groups(client):
    load(client, [text_event("a", 5.0, "one")])
    client.post(f"{BASE}/lectures/{L}/slides/{S}/group")
    assert client.get(f"{BASE}/lectures/{L}/groups").json()
    put(client, [seg(0, 1, 2, "hi")])
    assert client.get(f"{BASE}/lectures/{L}/groups").json() == []


def test_analyze_uses_stored_speech_instead_of_mock(client):
    load(client, [text_event("a", 100.0, "Kinetic energy")])
    put(client, [seg(0, 95, 99, "this is the real lecturer voice")])
    body = client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze").json()
    assert body["inputs_are_mock"]["speech"] is False
    summary = body["analyses"][0]["interpretation"]["summary"]
    assert "real lecturer voice" in summary
    # supplied speech still wins over stored speech
    own = {"speech": [seg(0, 99, 101, "request supplied words")]}
    again = client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze", json=own).json()
    assert "request supplied words" in again["analyses"][0]["interpretation"]["summary"]


def test_analyze_without_any_speech_still_labels_mock(client):
    load(client, [text_event("a", 100.0, "x")])
    body = client.post(f"{BASE}/lectures/{L}/slides/{S}/analyze").json()
    assert body["inputs_are_mock"]["speech"] is True


# ------------------------------------------------------------------ report


def test_report_shows_what_was_said_next_to_each_item(client):
    load(client, [text_event("a", 100.0, "Latency"), text_event("b", 400.0, "Throughput")])
    put(client, [seg(0, 98, 104, "latency is the delay"), seg(1, 399, 405, "throughput is volume")])
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "latency is the delay" in html
    assert "throughput is volume" in html
    assert html.count("Said:") == 2
    assert "Lecture transcript (2 segments)" in html


def test_report_escapes_hostile_speech_and_works_without_speech(client):
    load(client, [text_event("a", 100.0, "x")])
    put(client, [seg(0, 100, 101, "<script>alert(1)</script> & more")])
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp; more" in html
    other = "plain"
    client.post(
        f"{BASE}/events/import",
        json={
            "lecture_id": other,
            "slide_id": "s",
            "events": [text_event("p", 1.0, "p", page="s")],
        },
    )
    assert "Said:" not in client.get(f"{BASE}/lectures/{other}/report").text


def test_said_at_window_and_limits():
    speech = [
        SpeechSegment(id=i, start=float(i * 10), end=float(i * 10 + 5), text=f"s{i}")
        for i in range(12)
    ]
    assert said_at(None, 0, 5) == []
    assert said_at([], 0, 5) == []
    assert [s.text for s in said_at(speech, 50, 50)] == ["s4", "s5", "s6"]  # window 42..62
    assert len(said_at(speech, 0, 1000)) == MAX_QUOTES
    assert said_at([SpeechSegment(id=0, start=0, end=1, text="  ")], 0, 1) == []
    far = [SpeechSegment(id=0, start=500, end=510, text="late")]
    assert said_at(far, 0, 5) == []


def test_said_at_prefers_sentences_that_share_the_items_words():
    speech = [
        SpeechSegment(id=0, start=95, end=99, text="and transactions per day"),
        SpeechSegment(id=1, start=100, end=104, text="lock contention means two users wait"),
        SpeechSegment(id=2, start=105, end=108, text="think about the weather"),
    ]
    target = {"lock", "contention"}
    assert [s.text for s in said_at(speech, 100, 100, target)] == [
        "lock contention means two users wait"
    ]
    # no item words given: plain time order, as before
    assert len(said_at(speech, 100, 100)) == 3
    # nothing shares a word: falls back to the earliest segments in the window
    assert [s.id for s in said_at(speech, 100, 100, {"zebra"})] == [0, 1, 2]
