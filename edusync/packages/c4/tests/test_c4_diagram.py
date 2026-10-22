"""Demo Step 3 tests: diagrams (boxes joined by arrows) and structure-aware grouping.

All data is SYNTHETIC, shaped like the real BBB shapes seen in the rehearsal (an arrow whose end
is a `binding` to a shape and whose start is a free point next to the other shape).
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT, AnnotationRepository
from edusync_c4.diagram import build_diagrams, structural_links
from edusync_c4.grouping import group_timeline
from edusync_c4.report import SlideView, build_report_html
from edusync_c4.router import get_repository
from edusync_c4.schemas import GroupingConfig
from edusync_c4.timeline import build_timeline

BASE = "/api/c4"


def shape(sid, type_, ts, props, x=0.0, y=0.0):
    return {"id": sid, "type": type_, "x": x, "y": y, "timestamp": ts, "props": props}


def box(sid, ts, x, y, w=180.0, h=130.0, **props):
    return shape(sid, "geo", ts, {"geo": "rectangle", "w": w, "h": h, **props}, x, y)


def label(sid, ts, text, x, y, w=80.0, h=30.0):
    return shape(sid, "text", ts, {"text": text, "w": w, "h": h}, x, y)


def arrow(sid, ts, x, y, start, end, **props):
    return shape(sid, "arrow", ts, {"start": start, "end": end, **props}, x, y)


def bound(shape_id, ax=0.5, ay=0.5):
    return {"type": "binding", "boundShapeId": shape_id, "normalizedAnchor": {"x": ax, "y": ay}}


def point(x=0.0, y=0.0):
    return {"type": "point", "x": x, "y": y}


def timeline(events):
    return build_timeline(events, lecture_id="L", slide_id="S")


def diagrams(events, at=10_000.0):
    return build_diagrams(timeline(events).get_active_annotations(at))


def real_like(spread=60.0):
    """Two labelled boxes and an arrow: start is a loose point on box A, end is bound to box B."""
    return [
        box("A", 10, 400, 540),
        box("B", 10 + spread, 730, 540, w=190, h=120),
        label("tA", 10 + 2 * spread, "help", 430, 560),
        label("tB", 10 + 3 * spread, "needed", 760, 565),
        arrow("ar", 10 + 4 * spread, 580, 605, point(0, 0), bound("B", 0.015, 0.5)),
    ]


def test_real_like_diagram_is_read_as_a_sentence():
    (d,) = diagrams(real_like())
    assert d.sentences == ["help → needed"]
    assert [n.name for n in d.nodes] == ["help", "needed"]
    (edge,) = d.edges
    assert (edge.source_name, edge.target_name, edge.relation) == ("help", "needed", "directed")
    assert d.dangling_arrow_ids == []
    assert {"A", "B", "tA", "tB", "ar"} == set(d.shape_ids)  # boxes, labels and the arrow
    assert d.first_seconds == 10.0


@pytest.mark.parametrize(
    ("props", "sentence"),
    [
        ({}, "A1 → B1"),
        ({"arrowheadEnd": "arrow", "arrowheadStart": "arrow"}, "A1 ↔ B1"),
        ({"arrowheadEnd": "none", "arrowheadStart": "arrow"}, "B1 → A1"),
        ({"arrowheadEnd": "none", "arrowheadStart": "none"}, "A1 — B1"),
        ({"text": "causes"}, "A1 → B1 (causes)"),
        ({"text": "  "}, "A1 → B1"),
    ],
)
def test_arrow_direction_and_label(props, sentence):
    events = [
        box("a", 1, 0, 0, w=100, h=60, text="A1"),
        box("b", 2, 400, 0, w=100, h=60, text="B1"),
        arrow("ar", 3, 0, 0, bound("a"), bound("b"), **props),
    ]
    assert [s for d in diagrams(events) for s in d.sentences] == [sentence]


def test_arrow_bound_to_a_label_attaches_to_its_box():
    events = [
        box("a", 1, 0, 0),
        label("la", 2, "Mass", 20, 20),
        box("b", 3, 500, 0),
        label("lb", 4, "Energy", 520, 20),
        arrow("ar", 5, 0, 0, bound("la"), bound("lb")),
    ]
    (d,) = diagrams(events)
    assert d.sentences == ["Mass → Energy"]
    assert len(d.nodes) == 2  # the labels are not extra nodes


def test_loose_arrow_end_snaps_only_when_close():
    near = [
        box("a", 1, 0, 0, w=100, h=60, text="A1"),
        box("b", 2, 400, 0, w=100, h=60, text="B1"),
        arrow("ar", 3, 100, 30, point(0, 0), point(270, 0)),  # ends 30 px left of box B
    ]
    assert [s for d in diagrams(near) for s in d.sentences] == ["A1 → B1"]
    far = [*near[:2], arrow("ar", 3, 100, 30, point(0, 0), point(200, 0))]  # 100 px away
    assert diagrams(far) == []


def test_free_text_can_be_a_node():
    events = [
        label("m", 1, "Mass", 0, 0, w=100, h=30),
        label("e", 2, "Energy", 400, 0, w=100, h=30),
        arrow("ar", 3, 100, 15, point(0, 0), point(300, 0)),
    ]
    (d,) = diagrams(events)
    assert d.sentences == ["Mass → Energy"]


def test_two_diagrams_are_separate_and_ordered_by_time():
    events = [
        box("a1", 1, 0, 0, w=100, h=60, text="P"),
        box("b1", 2, 300, 0, w=100, h=60, text="Q"),
        arrow("r1", 3, 0, 0, bound("a1"), bound("b1")),
        box("a2", 100, 0, 500, w=100, h=60, text="X"),
        box("b2", 101, 300, 500, w=100, h=60, text="Y"),
        arrow("r2", 102, 0, 0, bound("a2"), bound("b2")),
    ]
    first, second = diagrams(events)
    assert (first.sentences, second.sentences) == (["P → Q"], ["X → Y"])
    assert first.diagram_id != second.diagram_id
    assert [d.diagram_id for d in diagrams(events)] == [first.diagram_id, second.diagram_id]


def test_chain_of_three_is_one_diagram_with_two_sentences():
    events = [
        box("a", 1, 0, 0, w=100, h=60, text="A"),
        box("b", 2, 300, 0, w=100, h=60, text="B"),
        box("c", 3, 600, 0, w=100, h=60, text="C"),
        arrow("r1", 4, 0, 0, bound("a"), bound("b")),
        arrow("r2", 5, 0, 0, bound("b"), bound("c")),
    ]
    (d,) = diagrams(events)
    assert d.sentences == ["A → B", "B → C"]


def test_lone_boxes_and_empty_input_are_not_diagrams():
    assert diagrams([box("a", 1, 0, 0, text="Solo")]) == []
    assert build_diagrams([]) == []


def test_unlabelled_node_gets_a_plain_name():
    events = [
        box("a", 1, 0, 0, w=100, h=60),
        box("b", 2, 300, 0, w=100, h=60, text="B"),
        arrow("r", 3, 0, 0, bound("a"), bound("b")),
    ]
    (d,) = diagrams(events)
    assert d.sentences == ["(unlabelled box) → B"]


def test_dangling_arrow_is_reported_not_drawn_as_a_link():
    events = [
        box("a", 1, 0, 0, w=100, h=60, text="A"),
        box("b", 2, 300, 0, w=100, h=60, text="B"),
        arrow("good", 3, 0, 0, bound("a"), bound("b")),
        arrow("loose", 4, 0, 0, bound("a"), point(900, 900)),
    ]
    (d,) = diagrams(events)
    assert d.sentences == ["A → B"]
    assert d.dangling_arrow_ids == ["loose"]


def test_huge_box_does_not_swallow_the_slide():
    events = [
        box("frame", 1, 0, 0, w=1400, h=800),
        label("m", 2, "Mass", 100, 100, w=100, h=30),
        label("e", 3, "Energy", 600, 100, w=100, h=30),
        arrow("r", 4, 0, 0, bound("m"), bound("e")),
    ]
    (d,) = diagrams(events)
    assert d.sentences == ["Mass → Energy"]


def test_uses_the_latest_position_of_a_moved_box():
    events = [
        box("a", 1, 0, 0, w=100, h=60, text="A"),
        box("b", 2, 300, 0, w=100, h=60, text="B"),
        box("b", 50, 300, 700, w=100, h=60, text="B"),  # B moved far below
        arrow("r", 51, 100, 30, point(0, 0), point(200, 0)),  # points at B's OLD place
    ]
    assert diagrams(events) == []


def test_deleted_shapes_do_not_form_diagrams():
    events = [
        box("a", 1, 0, 0, w=100, h=60, text="A"),
        box("b", 2, 300, 0, w=100, h=60, text="B"),
        arrow("r", 3, 0, 0, bound("a"), bound("b")),
        {"id": "r", "timestamp": 20, "is_deleted": True},
    ]
    assert len(diagrams(events, at=10)) == 1
    assert diagrams(events, at=30) == []


def test_hostile_names_stay_plain_text_in_the_report():
    events = [
        box("a", 1, 0, 0, w=100, h=60, text="<script>alert(1)</script>"),
        box("b", 2, 300, 0, w=100, h=60, text="B"),
        arrow("r", 3, 0, 0, bound("a"), bound("b")),
    ]
    tl = timeline(events)
    html = build_report_html("L", [SlideView("S", None, None, tl, group_timeline(tl))])
    assert "<h3>Diagrams</h3>" in html
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt; → B" in html


# ------------------------------------------------------------------ structure-aware grouping


def test_structural_links_pair_labels_boxes_and_arrow_ends():
    tl = timeline(real_like())
    links = set(structural_links(tl.get_active_annotations(10_000)))
    assert ("tA", "A") in links
    assert ("tB", "B") in links
    assert ("ar", "B") in links
    assert ("ar", "A") in links  # the loose start snaps to box A


def test_a_diagram_drawn_slowly_is_one_group():
    tl = timeline(real_like(spread=60.0))  # 4 minutes between first and last piece
    (g,) = group_timeline(tl)
    assert sorted(g.shape_ids) == ["A", "B", "ar", "tA", "tB"]
    assert g.native_text == "help / needed"
    assert g.method == "spatiotemporal-structural-v3"


def test_structural_grouping_can_be_switched_off():
    tl = timeline(real_like(spread=60.0))
    groups = group_timeline(tl, GroupingConfig(structural_links=False))
    assert len(groups) > 1


def test_unrelated_shape_stays_out_of_the_diagram_group():
    events = [*real_like(), shape("far", "text", 20, {"text": "aside", "w": 80, "h": 30}, 100, 20)]
    groups = group_timeline(timeline(events))
    assert sorted(len(g.shape_ids) for g in groups) == [1, 5]


def test_label_inside_a_box_groups_even_far_apart_in_time():
    events = [box("a", 1, 0, 0, w=300, h=200), label("t", 500, "late label", 50, 50)]
    (g,) = group_timeline(timeline(events))
    assert sorted(g.shape_ids) == ["a", "t"]


# ------------------------------------------------------------------ API


@pytest.fixture
def client(tmp_path):
    repo = AnnotationRepository(tmp_path / "c4" / "annotations.db")
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def _load(client, slide):
    events = [{**e, "pageId": slide} for e in real_like()]
    body = {"lecture_id": "L", "slide_id": slide, "events": events}
    assert client.post(f"{BASE}/events/import", json=body).status_code == 200


def test_diagrams_endpoint_with_a_slash_slide_id(client):
    slide = "pres/2"
    _load(client, slide)
    resp = client.get(f"{BASE}/lectures/L/slides/{slide}/diagrams")
    assert resp.status_code == 200
    (d,) = resp.json()
    assert d["sentences"] == ["help → needed"]
    assert client.get(f"{BASE}/lectures/L/slides/{slide}/diagrams?timestamp=5").json() == []
    assert client.get(f"{BASE}/lectures/L/slides/{slide}/diagrams?timestamp=-1").status_code == 422
    assert client.get(f"{BASE}/lectures/L/slides/nope/diagrams").status_code == 404


def test_report_endpoint_shows_the_diagram(client):
    _load(client, "pres/2")
    html = client.get(f"{BASE}/lectures/L/report").text
    assert "<h3>Diagrams</h3>" in html
    assert "help → needed" in html
