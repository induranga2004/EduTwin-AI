"""Underline / circle recognition and pointing at slide words. All strokes are SYNTHETIC but shaped
like real BBB strokes (a flat underline, a double-looped circle) from a recorded test lecture.
"""

import math
import random

import pytest

from edusync_c4.pointing import (
    circled_words,
    describe_marking,
    underlined_words,
)
from edusync_c4.schemas import (
    BoundingBox,
    NormalizedAnnotation,
    ShapeType,
    SlideWord,
    StrokeKind,
    StrokePoint,
)
from edusync_c4.strokes import classify_stroke, polygon_area


def pts(coords):
    return [StrokePoint(x=x, y=y) for x, y in coords]


def line(x0, y0, x1, y1, n=40, jitter=0.0, seed=1):
    rng = random.Random(seed)
    return pts(
        (
            x0 + (x1 - x0) * i / (n - 1) + rng.uniform(-jitter, jitter),
            y0 + (y1 - y0) * i / (n - 1) + rng.uniform(-jitter, jitter),
        )
        for i in range(n)
    )


def ellipse(cx, cy, rx, ry, loops=1.0, n=120, jitter=0.0, seed=2):
    rng = random.Random(seed)
    return pts(
        (
            cx + rx * math.cos(2 * math.pi * loops * i / (n - 1)) + rng.uniform(-jitter, jitter),
            cy + ry * math.sin(2 * math.pi * loops * i / (n - 1)) + rng.uniform(-jitter, jitter),
        )
        for i in range(n)
    )


def kind(points):
    return classify_stroke(points)[0]


# ------------------------------------------------------------------ classifier


def test_ideal_and_wobbly_underlines():
    assert kind(line(163, 225, 314, 225)) == StrokeKind.UNDERLINE  # like the real one: 151 x 0
    assert kind(line(100, 300, 420, 303, jitter=3.0)) == StrokeKind.UNDERLINE
    assert kind(line(420, 300, 100, 298)) == StrokeKind.UNDERLINE  # drawn right to left


def test_tilted_lines():
    assert kind(line(100, 300, 400, 340)) == StrokeKind.UNDERLINE  # about 7.6 degrees
    assert kind(line(100, 300, 400, 400)) == StrokeKind.LINE  # about 18 degrees
    assert kind(line(100, 100, 100, 300)) == StrokeKind.LINE  # vertical
    assert kind(line(100, 100, 300, 300)) == StrokeKind.LINE  # diagonal


def test_circles_ovals_and_double_loops():
    assert kind(ellipse(300, 300, 60, 60, loops=1.02)) == StrokeKind.CIRCLE
    assert kind(ellipse(260, 277, 106, 38, loops=1.0)) == StrokeKind.CIRCLE  # like the real 212x76
    assert kind(ellipse(260, 277, 106, 38, loops=2.1, jitter=2.0)) == StrokeKind.CIRCLE  # two loops
    assert kind(ellipse(300, 300, 60, 60, loops=0.93)) == StrokeKind.CIRCLE  # nearly closed


def test_open_arc_and_zigzag_are_scribbles():
    assert kind(ellipse(300, 300, 60, 60, loops=0.6)) == StrokeKind.SCRIBBLE  # a "C"
    zig = pts((i * 10, 30 if i % 2 else 0) for i in range(21))
    assert kind(zig) == StrokeKind.SCRIBBLE


def test_figure_eight_is_not_a_circle():
    n = 120
    eight = pts(
        (
            200 + 60 * math.sin(2 * math.pi * i / (n - 1)),
            300 + 40 * math.sin(4 * math.pi * i / (n - 1)),
        )
        for i in range(n)
    )
    assert kind(eight) == StrokeKind.SCRIBBLE


def test_boxy_closed_shape_is_a_loop():
    square = pts(
        [(100 + i * 5, 100) for i in range(21)]
        + [(200, 100 + i * 5) for i in range(1, 21)]
        + [(200 - i * 5, 200) for i in range(1, 21)]
        + [(100, 200 - i * 5) for i in range(1, 21)]
    )
    assert kind(square) == StrokeKind.LOOP


def test_dots_and_tiny_strokes():
    assert kind(pts([(100, 100)])) == StrokeKind.DOT
    assert kind(pts([(100, 100), (101, 100)])) == StrokeKind.DOT
    assert kind(line(100, 100, 108, 100, n=10)) == StrokeKind.DOT
    assert kind([]) == StrokeKind.DOT


def test_measurements_are_reported():
    _, m = classify_stroke(line(163, 225, 314, 225))
    assert m["width"] == 151.0
    assert m["straightness"] >= 0.99
    assert "angle" in m
    _, c = classify_stroke(ellipse(300, 300, 60, 60))
    assert "fill" in c
    assert 0.6 <= c["fill"] <= 0.9


def test_polygon_area():
    assert polygon_area(pts([(0, 0), (10, 0), (10, 10), (0, 10)])) == 100.0


# ------------------------------------------------------------------ pointing at words

# Boxes shaped like the real slide 3 (PDF points x 1.5 = whiteboard units)
WORDS = [
    SlideWord(text="▪", x=130.5, y=193.7, width=13.8, height=33.2),
    SlideWord(text="Throughput", x=157.5, y=187.9, width=157.5, height=39.5),
    SlideWord(text="\u2013", x=323.5, y=187.9, width=13.4, height=39.5),
    SlideWord(text="Transactions", x=351.9, y=187.9, width=174.0, height=39.5),
    SlideWord(text="per", x=534.4, y=187.9, width=44.7, height=39.5),
    SlideWord(text="second", x=587.9, y=187.9, width=100.3, height=39.5),
    SlideWord(text="▪", x=130.5, y=253.5, width=13.8, height=33.2),
    SlideWord(text="Response", x=157.5, y=246.3, width=132.5, height=40.5),
    SlideWord(text="Time", x=300.0, y=246.3, width=68.0, height=40.5),
    SlideWord(text="\u2013", x=379.0, y=246.3, width=13.4, height=40.5),
    SlideWord(text="Time", x=395.0, y=246.3, width=68.0, height=40.5),
    SlideWord(text="per", x=470.0, y=246.3, width=44.7, height=40.5),
    SlideWord(text="query", x=525.0, y=246.3, width=90.0, height=40.5),
]


def names(words):
    return " ".join(w.text for w in words)


def test_underline_names_the_word_above_it():
    stroke = line(163, 225, 314, 225)  # the real underline
    assert names(underlined_words(stroke, WORDS)) == "Throughput"


def test_underline_over_two_words_and_in_reading_order():
    stroke = line(352, 230, 580, 232)
    assert names(underlined_words(stroke, WORDS)) == "Transactions per"


def test_underline_needs_to_cover_most_of_a_word():
    stroke = line(163, 226, 440, 226)  # Throughput, the dash and most of Transactions
    assert names(underlined_words(stroke, WORDS)) == "Throughput Transactions"
    short = line(163, 226, 200, 226)  # only a quarter of "Throughput"
    assert underlined_words(short, WORDS) == []


def test_underline_picks_only_the_closest_line():
    under_line_two = line(160, 292, 365, 292)
    assert names(underlined_words(under_line_two, WORDS)) == "Response Time"


def test_underline_with_nothing_above_it_names_nothing():
    assert underlined_words(line(163, 400, 314, 400), WORDS) == []
    assert underlined_words(line(163, 225, 314, 225), []) == []


def test_circle_names_the_enclosed_words_only():
    loop = ellipse(260, 277, 106, 38, loops=1.05)  # around Response + Time
    assert names(circled_words(loop, WORDS)) == "Response Time"


def test_circle_ignores_bullets_and_grazed_words():
    wide = ellipse(300, 267, 190, 36, loops=1.0)  # includes the bullet and reaches the dash
    result = names(circled_words(wide, WORDS))
    assert "▪" not in result
    assert result.startswith("Response Time")
    grazing = ellipse(520, 300, 60, 25, loops=1.0)  # mostly below the text
    assert circled_words(grazing, WORDS) == []


def test_circle_around_two_lines_reads_top_to_bottom():
    big = ellipse(240, 250, 200, 90, loops=1.0)
    assert names(circled_words(big, WORDS)).startswith("Throughput")
    assert "Response" in names(circled_words(big, WORDS))


def test_circle_drawn_twice_and_either_direction_still_names_the_words():
    twice = ellipse(260, 277, 106, 38, loops=2.1, n=332, jitter=1.5)
    assert names(circled_words(twice, WORDS)) == "Response Time"
    backwards = list(reversed(ellipse(260, 277, 106, 38, loops=1.05)))
    assert names(circled_words(backwards, WORDS)) == "Response Time"


def test_circle_around_empty_space():
    assert circled_words(ellipse(900, 600, 60, 40), WORDS) == []


# ------------------------------------------------------------------ describe_marking


def annotation(points, shape_type=ShapeType.DRAW):
    xs = [p.x for p in points] or [0.0]
    ys = [p.y for p in points] or [0.0]
    return NormalizedAnnotation(
        event_id="e",
        lecture_id="l",
        slide_id="s",
        shape_id="shape:1",
        shape_type=shape_type,
        timestamp_seconds=1.0,
        bounds=BoundingBox(x=min(xs), y=min(ys), width=max(xs) - min(xs), height=max(ys) - min(ys)),
        stroke_points=points,
    )


def test_describe_marking_underline_and_circle():
    under = describe_marking(annotation(line(163, 225, 314, 225)), WORDS)
    assert (under.kind, under.action, under.target_text) == (
        StrokeKind.UNDERLINE,
        "underlined",
        "Throughput",
    )
    circle = describe_marking(annotation(ellipse(260, 277, 106, 38)), WORDS)
    assert (circle.kind, circle.action, circle.target_text) == (
        StrokeKind.CIRCLE,
        "circled",
        "Response Time",
    )
    boxed = describe_marking(
        annotation(
            pts(
                [(150 + i * 10, 240) for i in range(25)]
                + [(390, 240 + i * 5) for i in range(1, 17)]
                + [(390 - i * 10, 320) for i in range(1, 25)]
                + [(150, 320 - i * 5) for i in range(1, 17)]
            )
        ),
        WORDS,
    )
    assert (boxed.kind, boxed.action) == (StrokeKind.LOOP, "boxed")


def test_describe_marking_without_words_or_for_other_shapes():
    plain = describe_marking(annotation(line(163, 225, 314, 225)), None)
    assert (plain.kind, plain.action, plain.target_text) == (StrokeKind.UNDERLINE, None, None)
    assert describe_marking(annotation([], ShapeType.TEXT), WORDS) is None
    assert describe_marking(annotation(line(163, 225, 314, 225), ShapeType.HIGHLIGHT), WORDS)


@pytest.mark.parametrize("kind_", [StrokeKind.SCRIBBLE, StrokeKind.LINE, StrokeKind.DOT])
def test_other_strokes_name_nothing(kind_):
    sample = {
        StrokeKind.SCRIBBLE: pts((i * 10, 30 if i % 2 else 0) for i in range(21)),
        StrokeKind.LINE: line(100, 100, 100, 300),
        StrokeKind.DOT: pts([(100, 100)]),
    }[kind_]
    marking = describe_marking(annotation(sample), WORDS)
    assert marking.kind == kind_
    assert marking.action is None


# ------------------------------------------------------------------ API, report and explanations

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from edusync_c4 import COMPONENT, AnnotationRepository, explain_slide  # noqa: E402
from edusync_c4.grouping import group_timeline  # noqa: E402
from edusync_c4.router import get_repository  # noqa: E402
from edusync_c4.schemas import SpeechSegment  # noqa: E402
from edusync_c4.timeline import build_timeline  # noqa: E402

BASE = "/api/c4"
L, S = "lec", "pres/3"
WORD_JSON = [w.model_dump() for w in WORDS]


def stroke_event(shape, ts, points):
    segment = {"points": [{"x": p.x, "y": p.y, "z": 0.5} for p in points]}
    return {
        "id": shape,
        "type": "draw",
        "x": 0,
        "y": 0,
        "timestamp": ts,
        "props": {"segments": [segment]},
        "pageId": S,
    }


def lecture_events():
    return [
        stroke_event("under", 172, line(163, 225, 314, 225, n=93)),
        stroke_event("circle", 200, ellipse(260, 277, 106, 38, loops=2.1, n=332, jitter=1.5)),
    ]


@pytest.fixture
def client(tmp_path):
    repo = AnnotationRepository(tmp_path / "c4" / "annotations.db")
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def load(client, with_words=True, speech=None):
    body = {"lecture_id": L, "slide_id": S, "events": lecture_events()}
    assert client.post(f"{BASE}/events/import", json=body).status_code == 200
    if with_words:
        resp = client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": WORD_JSON})
        assert resp.status_code == 200
    if speech:
        client.put(f"{BASE}/lectures/{L}/speech", json={"segments": speech, "offset_seconds": 0})


def test_words_endpoint_roundtrip_and_validation(client):
    resp = client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": WORD_JSON})
    assert resp.json() == {"words": len(WORDS)}
    stored = client.get(f"{BASE}/lectures/{L}/slides/{S}/words").json()
    assert stored[1]["text"] == "Throughput"
    assert client.get(f"{BASE}/lectures/{L}/slides/other/words").json() == []
    bad = {"text": "", "x": 0, "y": 0, "width": 1, "height": 1}
    assert (
        client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": [bad]}).status_code
        == 422
    )
    neg = {"text": "a", "x": 0, "y": 0, "width": -1, "height": 1}
    assert (
        client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": [neg]}).status_code
        == 422
    )
    many = [{"text": "a", "x": 0, "y": 0, "width": 1, "height": 1}] * 5001
    assert (
        client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": many}).status_code == 422
    )


def test_words_route_does_not_break_the_plain_slide_route(client):
    assert client.put(f"{BASE}/lectures/{L}/slides/{S}", json={"number": 3}).json()["slide_id"] == S
    client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": WORD_JSON})
    assert [s["slide_id"] for s in client.get(f"{BASE}/lectures/{L}/slides").json()] == [S]


def test_new_words_invalidate_cached_groups(client):
    load(client, with_words=False)
    client.post(f"{BASE}/lectures/{L}/slides/{S}/group")
    assert client.get(f"{BASE}/lectures/{L}/groups").json()
    client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": WORD_JSON})
    assert client.get(f"{BASE}/lectures/{L}/groups").json() == []


def test_delete_lecture_removes_words(client):
    load(client)
    assert client.delete(f"{BASE}/lectures/{L}").json()["slide_words"] == 1
    assert client.get(f"{BASE}/lectures/{L}/slides/{S}/words").json() == []


def test_report_names_the_underline_and_the_circle(client):
    load(client)
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "Underline under “Throughput”" in html
    assert "Circle around “Response Time”" in html
    assert "the lecturer underlined “Throughput”" in html
    assert "the lecturer circled “Response Time”" in html
    assert "Stroke measured as underline" in html
    assert "not recognised" not in html


def test_repeated_circle_is_one_phrase_in_time_order(client):
    events = [
        stroke_event("circle2", 188, ellipse(260, 277, 106, 38, loops=1.05)),
        stroke_event("circle1", 180, ellipse(260, 277, 106, 38, loops=1.05)),
        stroke_event("under", 172, line(163, 225, 314, 225, n=93)),
    ]
    body = {"lecture_id": L, "slide_id": S, "events": events}
    assert client.post(f"{BASE}/events/import", json=body).status_code == 200
    client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": WORD_JSON})
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "underlined “Throughput” and circled “Response Time” twice" in html
    assert html.count("circled “Response Time”") == 1
    times_two = "\N{MULTIPLICATION SIGN}2"
    assert f"Underline under “Throughput”; Circle around “Response Time” {times_two}" in html


def test_report_without_words_still_recognises_the_kind(client):
    load(client, with_words=False)
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "<b>Underline</b>" in html
    assert "<b>Circle</b>" in html
    assert "drew an underline (no slide text was found there)" in html
    assert "under “" not in html


def test_hostile_word_text_is_escaped_in_the_report(client):
    load(client, with_words=False)
    evil = [
        dict(w, text="<script>x</script>") if w["text"] == "Throughput" else w for w in WORD_JSON
    ]
    client.put(f"{BASE}/lectures/{L}/slides/{S}/words", json={"words": evil})
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "<script>x" not in html
    assert "&lt;script&gt;x&lt;/script&gt;" in html


def test_explain_uses_marked_words_to_choose_what_was_said():
    tl = build_timeline(lecture_events(), lecture_id=L, slide_id=S)
    groups = group_timeline(tl)
    speech = [
        SpeechSegment(id=0, start=170, end=173, text="throughput is how much work gets done"),
        SpeechSegment(id=1, start=174, end=177, text="please remember to submit the form"),
    ]
    out = explain_slide(tl, groups, speech, None, words=WORDS)
    by_action = {a.recognized.markings[0].action: a for a in out}
    summary = by_action["underlined"].interpretation.summary
    assert "underlined “Throughput”" in summary
    assert "throughput is how much work gets done" in summary
    assert "remember to submit" not in summary
    assert by_action["underlined"].interpretation.category == "marking"


def test_two_marks_in_one_group_are_both_named():
    near = [
        stroke_event("a", 10, line(163, 225, 314, 225, n=40)),
        stroke_event("b", 12, ellipse(260, 277, 106, 38, loops=1.0, n=120)),
    ]
    tl = build_timeline(near, lecture_id=L, slide_id=S)
    groups = group_timeline(tl)
    out = explain_slide(tl, groups, None, None, words=WORDS)
    summaries = " ".join(a.interpretation.summary for a in out)
    assert "underlined “Throughput”" in summaries
    assert "circled “Response Time”" in summaries
