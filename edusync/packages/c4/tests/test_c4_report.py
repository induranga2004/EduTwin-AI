"""Demo Step 2 tests: the lecture report. All data is SYNTHETIC (shaped like real BBB shapes)."""

import base64
import re
import xml.etree.ElementTree as ET

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT, AnnotationRepository
from edusync_c4.grouping import group_timeline
from edusync_c4.report import SlideView, build_report_html, clock, render_board, wrap_text
from edusync_c4.router import get_repository
from edusync_c4.schemas import SlideInfo
from edusync_c4.timeline import build_timeline

L, S = "lec", "pres/2"
BASE = "/api/c4"
SLIDE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540"><rect/></svg>'


def shape(sid, type_, ts, props, x=100, y=100, **extra):
    return {"id": sid, "type": type_, "x": x, "y": y, "timestamp": ts, "props": props, **extra}


def view(events, svg=SLIDE_SVG, info=None):
    tl = build_timeline(events, lecture_id=L, slide_id=S)
    return SlideView(S, info, svg, tl, group_timeline(tl))


def board(events, at=1000.0, size=(1440.0, 810.0), **kw):
    return render_board(view(events, **kw), at, size)[0]


def parse_board(html):
    match = re.search(r'<svg class="board".*?</svg>', html, re.S)
    assert match, "no board svg found"
    return ET.fromstring(match.group(0))  # raises if not well-formed XML


STROKE = {"segments": [{"type": "free", "points": [{"x": 0, "y": 0}, {"x": 40, "y": 10}]}]}


def test_clock_and_wrap_text():
    assert clock(5) == "00:05"
    assert clock(194.4) == "03:14"
    assert clock(3725) == "1:02:05"
    assert clock(-3) == "00:00"
    lines = wrap_text("one two three four five six seven", 100, 20)
    assert len(lines) > 1
    assert all(len(line) <= 9 for line in lines)
    assert wrap_text("a\n\nb", 500, 20) == ["a", "", "b"]
    assert wrap_text("x" * 50, 100, 20)  # an unbreakable word is still split


def test_board_is_well_formed_and_uses_slide_size():
    events = [
        shape("d", "draw", 1, STROKE),
        shape("t", "text", 2, {"text": "Kinetic energy", "size": "m", "w": 300}),
        shape("g", "geo", 3, {"geo": "ellipse", "w": 100, "h": 60, "text": "Box", "fill": "semi"}),
        shape("n", "note", 4, {"text": "hello", "color": "yellow"}),
    ]
    root = parse_board(board(events, size=(1440.0, 810.0)))
    assert root.attrib["viewBox"] == "0 0 1440 810"
    text = ET.tostring(root, encoding="unicode")
    assert "polyline" in text
    assert "ellipse" in text
    assert "Kinetic energy" in text


def test_slide_svg_is_embedded_as_data_uri_not_inline():
    risky = '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    out = board([shape("t", "text", 1, {"text": "x"})], svg=risky)
    assert "<script" not in out
    match = re.search(r'href="data:image/svg\+xml;base64,([^"]+)"', out)
    assert match
    assert base64.b64decode(match.group(1)).decode() == risky


def test_missing_slide_image_falls_back_to_plain_background():
    out = board([shape("t", "text", 1, {"text": "x"})], svg=None)
    assert "<image" not in out
    assert "#f4f4f4" in out


def test_lecture_text_is_escaped_everywhere():
    evil = '<script>alert("x")</script> & "q"'
    events = [
        shape("t", "text", 1, {"text": evil}),
        shape("n", "note", 2, {"text": evil}),
        shape("g", "geo", 3, {"geo": "rectangle", "w": 80, "h": 40, "text": evil}),
    ]
    info = SlideInfo(lecture_id=L, slide_id=S, number=2, text=evil)
    html = build_report_html("<b>id</b>", [view(events, info=info)])
    assert "<script" not in html
    assert "<b>id</b>" not in html
    assert "&lt;script&gt;" in html
    parse_board(html)  # still well-formed with hostile text


def test_long_text_wraps_and_note_grows():
    long_text = "word " * 120
    out = board([shape("n", "note", 1, {"text": long_text.strip()})])
    root = parse_board(out)
    tspans = [e for e in root.iter() if e.tag.endswith("tspan")]
    assert len(tspans) > 10
    rect = next(e for e in root.iter() if e.tag.endswith("rect") and e.attrib.get("rx") == "4")
    assert float(rect.attrib["height"]) > 200  # grew beyond the default sticky note


def test_deleted_shapes_are_not_drawn_after_removal_but_listed_as_removed():
    events = [
        shape("a", "text", 10, {"text": "keep me"}),
        shape("b", "text", 20, {"text": "remove me"}),
        {"id": "b", "timestamp": 90, "is_deleted": True},
    ]
    html = build_report_html(L, [view(events)], at_seconds=95)
    drawn = ET.tostring(parse_board(html), encoding="unicode")
    assert "keep me" in drawn
    assert "remove me" not in drawn
    assert "removed at 01:30" in html


def test_at_parameter_shows_the_board_at_that_time():
    events = [
        shape("a", "text", 10, {"text": "first"}),
        shape("b", "text", 50, {"text": "second"}),
    ]
    early = build_report_html(L, [view(events)], at_seconds=20)
    late = build_report_html(L, [view(events)], at_seconds=60)
    assert "second" not in ET.tostring(parse_board(early), encoding="unicode")
    assert "second" in ET.tostring(parse_board(late), encoding="unicode")
    assert "Board at 00:20" in early
    assert early.count('<svg class="board"') == 1


def test_clear_is_listed_and_empties_the_board():
    events = [
        shape("a", "text", 10, {"text": "gone"}),
        {"eventName": "ClearWhiteboardEvent", "timestamp": 30},
    ]
    html = build_report_html(L, [view(events)], at_seconds=40)
    assert "Board cleared" in html
    assert "gone" not in ET.tostring(parse_board(html), encoding="unicode")


def test_arrows_attached_to_shapes_end_at_the_shape():
    events = [
        shape("a", "geo", 1, {"geo": "rectangle", "w": 100, "h": 50}, x=100, y=100),
        shape("b", "geo", 2, {"geo": "rectangle", "w": 100, "h": 50}, x=500, y=100),
        shape(
            "arrow",
            "arrow",
            3,
            {
                "start": {
                    "type": "binding",
                    "boundShapeId": "a",
                    "normalizedAnchor": {"x": 1, "y": 0.5},
                },
                "end": {
                    "type": "binding",
                    "boundShapeId": "b",
                    "normalizedAnchor": {"x": 0, "y": 0.5},
                },
                "arrowheadEnd": "arrow",
                "text": "causes",
            },
            x=0,
            y=0,
        ),
    ]
    root = parse_board(board(events))
    line = next(e for e in root.iter() if e.tag.endswith("line"))
    assert (float(line.attrib["x1"]), float(line.attrib["y1"])) == (200.0, 125.0)
    assert (float(line.attrib["x2"]), float(line.attrib["y2"])) == (500.0, 125.0)
    assert "causes" in ET.tostring(root, encoding="unicode")


def test_point_arrow_and_line_and_frame_and_unknown_render():
    events = [
        shape(
            "p", "arrow", 1, {"start": {"type": "point", "x": 0, "y": 0}, "end": {"x": 80, "y": 40}}
        ),
        shape(
            "l",
            "line",
            2,
            {
                "handles": {
                    "a": {"x": 0, "y": 0, "index": "a1"},
                    "b": {"x": 50, "y": 50, "index": "a2"},
                }
            },
        ),
        shape("f", "frame", 3, {"w": 200, "h": 100, "name": "Group A"}),
        shape("u", "mystery", 4, {"w": 10, "h": 10}),
    ]
    text = ET.tostring(parse_board(board(events)), encoding="unicode")
    assert "Group A" in text
    assert "stroke-dasharray" in text  # unknown shape shown as a dashed box


def test_rotation_and_opacity_are_applied():
    out = board([shape("t", "text", 1, {"text": "x"}, rotation=1.5707963, opacity=0.5)])
    assert "rotate(90)" in out
    assert 'opacity="0.5"' in out


def test_page_summary_slide_titles_and_order():
    info = SlideInfo(lecture_id=L, slide_id=S, number=2, text="High Availability\nmore")
    html = build_report_html(L, [view([shape("t", "text", 5, {"text": "x"})], info=info)])
    assert "Slide 2 &mdash; High Availability" in html
    assert "1 slide with annotations, 1 items" in html
    assert "<script" not in html
    assert "Content-Security-Policy" in html


def test_empty_lecture_message():
    assert "No annotations were captured" in build_report_html(L, [])


# ---------------------------------------------------------------------------- API


@pytest.fixture
def client(tmp_path):
    repo = AnnotationRepository(tmp_path / "c4" / "annotations.db")
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def _events(page):
    return [
        {**shape("a", "text", 10, {"text": "hello"}), "pageId": page},
        {**shape("b", "draw", 20, STROKE), "pageId": page},
    ]


def test_report_endpoint_serves_html_with_slides_in_order(client):
    for number, page in ((2, "pres/2"), (1, "pres/1")):
        client.post(
            f"{BASE}/events/import",
            json={"lecture_id": L, "slide_id": page, "events": _events(page)},
        )
        client.put(
            f"{BASE}/lectures/{L}/slides/{page}",
            json={"number": number, "scaled_width": 1440, "scaled_height": 810, "svg": SLIDE_SVG},
        )
    resp = client.get(f"{BASE}/lectures/{L}/report")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in resp.headers["content-security-policy"]
    body = resp.text
    assert body.index("Slide 1") < body.index("Slide 2")
    assert body.count('<svg class="board"') == 2
    assert client.get(f"{BASE}/lectures/{L}/report?at=15").status_code == 200


def test_report_endpoint_errors(client):
    assert client.get(f"{BASE}/lectures/nope/report").status_code == 404
    client.post(
        f"{BASE}/events/import", json={"lecture_id": L, "slide_id": S, "events": _events(S)}
    )
    assert client.get(f"{BASE}/lectures/{L}/report?at=-1").status_code == 422
    # works with no slide image or slide info stored at all
    assert "Slide" in client.get(f"{BASE}/lectures/{L}/report").text


def test_events_api_still_hides_shape_state(client):
    client.post(
        f"{BASE}/events/import", json={"lecture_id": L, "slide_id": S, "events": _events(S)}
    )
    hidden = client.get(f"{BASE}/lectures/{L}/events").json()
    assert all(e["shape_state"] == {} for e in hidden)
    shown = client.get(f"{BASE}/lectures/{L}/events", params={"include_raw": True}).json()
    assert any(e["shape_state"] for e in shown)


def test_default_shows_fullest_board_and_the_end_board_when_they_differ():
    events = [
        shape("a", "text", 10, {"text": "alpha"}),
        shape("b", "text", 20, {"text": "beta"}),
        {"eventName": "ClearWhiteboardEvent", "timestamp": 30},
        shape("c", "text", 40, {"text": "gamma"}),
    ]
    html = build_report_html(L, [view(events)])
    assert "Board at its fullest (00:20)" in html
    assert "Board at the end (00:40)" in html
    boards = re.findall(r'<svg class="board".*?</svg>', html, re.S)
    assert len(boards) == 2
    assert "alpha" in boards[0]
    assert "beta" in boards[0]
    assert "alpha" not in boards[1]
    assert "gamma" in boards[1]


def test_default_shows_one_board_when_the_end_is_the_fullest():
    events = [shape("a", "text", 10, {"text": "alpha"}), shape("b", "text", 20, {"text": "beta"})]
    html = build_report_html(L, [view(events)])
    assert html.count('<svg class="board"') == 1
    assert "Board at its fullest (00:20)" in html
    assert "Board at the end" not in html
