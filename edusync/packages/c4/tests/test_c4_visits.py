"""Slide visits: which slide was on screen when, and keeping speech on its own slide."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT, AnnotationRepository
from edusync_c4.local_explain import speech_on_slide
from edusync_c4.router import get_repository
from edusync_c4.schemas import SpeechSegment

BASE = "/api/c4"
L = "lec"
S2, S3 = "pres/2", "pres/3"


def seg(i, start, end, text):
    return SpeechSegment(id=f"s{i}", start=start, end=end, text=text)


@pytest.fixture
def repo(tmp_path):
    return AnnotationRepository(tmp_path / "c4" / "annotations.db")


@pytest.fixture
def client(repo):
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def seen(client, slide, seconds):
    return client.post(f"{BASE}/lectures/{L}/slides/{slide}/seen", json={"seconds": seconds})


def test_close_samples_make_one_visit_and_a_gap_makes_two(repo):
    for t in (10, 12, 14, 16):
        repo.record_slide_seen(L, S2, t)
    repo.record_slide_seen(L, S2, 60)  # more than 10 s later: back again, a new visit
    assert repo.get_slide_visits(L) == {S2: [(10, 16), (60, 60)]}


def test_changing_slide_closes_the_visit(repo):
    for slide, t in ((S2, 10), (S2, 12), (S3, 14), (S3, 16), (S2, 18)):
        repo.record_slide_seen(L, slide, t)
    assert repo.get_slide_visits(L) == {S2: [(10, 12), (18, 18)], S3: [(14, 16)]}


def test_repeated_and_late_samples_do_not_shrink_a_visit(repo):
    for t in (10, 14, 12, 14):
        repo.record_slide_seen(L, S2, t)
    assert repo.get_slide_visits(L) == {S2: [(10, 14)]}


def test_seen_endpoint_validates_and_visits_are_listed(client):
    assert seen(client, S2, 5).json() == {"recorded": True}
    seen(client, S3, 9)
    assert [
        (v["slide_id"], v["start_seconds"])
        for v in client.get(f"{BASE}/lectures/{L}/visits").json()
    ] == [
        (S2, 5),
        (S3, 9),
    ]
    assert seen(client, S2, -1).status_code == 422
    assert seen(client, S2, 1e12).status_code == 422
    assert client.post(f"{BASE}/lectures/{L}/slides/{S2}/seen", json={}).status_code == 422
    assert client.get(f"{BASE}/lectures/none/visits").json() == []


def test_speech_on_slide_drops_what_was_said_on_another_slide():
    speech = [
        seg(1, 0, 8, "intro"),
        seg(2, 100, 108, "about the diagram"),
        seg(3, 127, 135, "throughput is transactions per second"),
    ]
    slide2 = [(100.0, 124.0)]  # slide 3 appeared at about 125
    slide3 = [(125.0, 160.0)]
    assert [s.id for s in speech_on_slide(speech, slide2)] == ["s2"]
    assert [s.id for s in speech_on_slide(speech, slide3)] == ["s3"]


def test_speech_at_the_edge_of_a_visit_is_kept_on_both_sides():
    speech = [seg(1, 120, 130, "said while flipping")]
    assert speech_on_slide(speech, [(100.0, 121.0)])  # overlaps the end
    assert speech_on_slide(speech, [(129.0, 150.0)])  # overlaps the start


def test_without_visits_speech_is_unchanged():
    speech = [seg(1, 0, 5, "hello")]
    assert speech_on_slide(speech, None) is speech
    assert speech_on_slide(speech, []) is speech
    assert speech_on_slide(None, [(0.0, 1.0)]) is None


def test_delete_lecture_removes_visits(client, repo):
    seen(client, S2, 5)
    assert client.delete(f"{BASE}/lectures/{L}").json()["slide_visits"] == 1
    assert repo.get_slide_visits(L) == {}


def test_report_keeps_quotes_to_their_own_slide(client):
    def draw(slide, shape, ts, text):
        events = [
            {
                "id": shape,
                "type": "text",
                "x": 0,
                "y": 0,
                "timestamp": ts,
                "props": {"text": text, "w": 100, "h": 30},
            }
        ]
        body = {"lecture_id": L, "slide_id": slide, "events": events}
        assert client.post(f"{BASE}/events/import", json=body).status_code == 200

    draw(S2, "a", 100, "Latency")
    draw(S3, "b", 130, "Throughput")
    speech = [
        seg(1, 98, 104, "latency is the delay of a request"),
        seg(2, 126, 133, "throughput is requests finished per second"),
    ]
    body = {"segments": [s.model_dump() for s in speech], "offset_seconds": 0}
    assert client.put(f"{BASE}/lectures/{L}/speech", json=body).status_code == 200
    for t in (96, 100, 104, 108, 112, 116, 120):
        seen(client, S2, t)
    for t in (124, 128, 132, 136):
        seen(client, S3, t)
    html = client.get(f"{BASE}/lectures/{L}/report").text
    slides = html.split('<section class="slide">')[1:]
    assert len(slides) == 2
    first, second = (s.split("</section>")[0] for s in slides)
    assert "latency is the delay" in first
    assert "throughput is requests" not in first
    assert "throughput is requests" in second
    assert "latency is the delay" not in second
