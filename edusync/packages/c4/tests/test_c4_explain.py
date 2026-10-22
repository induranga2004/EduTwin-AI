"""Demo Step 5 tests: local, rule-based "What it means" summaries. All data is SYNTHETIC."""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from edusync_c4 import COMPONENT, AnnotationRepository, EvidenceInterpreter, explain_slide
from edusync_c4.grouping import group_timeline
from edusync_c4.local_explain import best_slide_line, rank_speech, terms
from edusync_c4.router import get_repository
from edusync_c4.schemas import (
    AnnotationContext,
    AnnotationInterpretation,
    GroupAnalysis,
    GroupingConfig,
    RecognizedContent,
    SpeechEvidence,
    SpeechSegment,
)
from edusync_c4.timeline import build_timeline

L, S = "lec", "pres/1"
BASE = "/api/c4"
SLIDE_TEXT = (
    "Database Performance\n"
    "▪ Latency is the delay between a request and its response\n"
    "▪ Throughput is how many requests finish per second\n"
    "▪ Backups protect data"
)


def shape(sid, type_, ts, props, x=100.0, y=100.0, **extra):
    return {"id": sid, "type": type_, "x": x, "y": y, "timestamp": ts, "props": props, **extra}


def text(sid, ts, words, x=100.0, y=100.0):
    return shape(sid, "text", ts, {"text": words, "w": 200, "h": 30}, x, y)


def speech_ev(i, start, end, words, relation="during"):
    return SpeechEvidence(
        evidence_id=f"speech:{i}",
        segment_id=i,
        start=start,
        end=end,
        text=words,
        relation=relation,
    )


def context(speech, native="Latency", diagram=None, start=100.0, end=105.0):
    return AnnotationContext(
        group_id="g1",
        lecture_id=L,
        slide_id=S,
        span_start=start,
        span_end=end,
        window_start=start - 90,
        window_end=end + 90,
        native_text=native,
        slide_evidence_id="slide:S",
        slide_text_excerpt=SLIDE_TEXT,
        speech=speech,
        diagram_sentences=diagram or [],
    )


def recognized(kind="native_text", text_="Latency", strokes=0):
    return RecognizedContent(
        group_id="g1",
        kind=kind,
        text=text_,
        freehand_stroke_count=strokes,
        recognizer="t",
        notes=[],
    )


# ------------------------------------------------------------------ word matching


def test_terms_drop_stop_words_and_stem_lightly():
    assert terms("The latency is a delay of queries") == {"latency", "delay", "query"}
    assert terms("") == set()
    assert terms("the and for") == set()
    assert terms("Backups backup") == {"backup"}
    assert terms("a query") == terms("queries")


def test_rank_speech_prefers_shared_words_over_time_closeness():
    ctx = context(
        [
            speech_ev(0, 99, 101, "let me move on quickly"),  # nearest in time, off-topic
            speech_ev(1, 120, 125, "latency is the delay of one request", "after"),
            speech_ev(2, 60, 64, "we also care about cost", "before"),
        ]
    )
    chosen = rank_speech(ctx, terms("Latency"))
    assert [s.evidence_id for s in chosen] == ["speech:1"]


def test_rank_speech_falls_back_to_closeness_when_no_word_matches():
    ctx = context(
        [
            speech_ev(0, 50, 55, "something far before", "before"),
            speech_ev(1, 101, 103, "this moment"),
            speech_ev(2, 140, 145, "later", "after"),
        ]
    )
    chosen = rank_speech(ctx, terms("zebra"), limit=2)
    assert [s.evidence_id for s in chosen] == ["speech:1", "speech:2"]
    assert rank_speech(context([]), terms("x")) == []
    assert rank_speech(context([speech_ev(0, 1, 2, "   ")]), terms("x")) == []


def test_best_slide_line_picks_the_matching_bullet():
    assert "Latency is the delay" in best_slide_line(SLIDE_TEXT, terms("latency"))
    assert "Throughput" in best_slide_line(SLIDE_TEXT, terms("throughput second"))
    assert best_slide_line(SLIDE_TEXT, terms("zebra")) is None
    assert best_slide_line(None, terms("latency")) is None
    assert best_slide_line(SLIDE_TEXT, set()) is None


# ------------------------------------------------------------------ the interpreter


def test_summary_for_written_text_quotes_speech_and_slide():
    ctx = context([speech_ev(1, 98, 104, "latency is how long one request waits")])
    out = EvidenceInterpreter().interpret(ctx, recognized())
    assert out.method == "evidence-local-v1"
    assert "At 01:40 the lecturer wrote “Latency”." in out.summary
    assert "latency is how long one request waits" in out.summary
    assert "Related slide text" in out.summary
    assert out.cited_evidence_ids == ["annotation:g1", "speech:1", "slide:S"]
    assert "not AI" in out.limitations[0]
    assert out.category == "native_text"


def test_summary_variants_by_content_kind():
    interp = EvidenceInterpreter()
    freehand = interp.interpret(
        context([], native=None), recognized("unrecognized_freehand", None, 3)
    ).summary
    assert "drew 3 freehand strokes (the drawing itself is not recognised)" in freehand
    mixed = interp.interpret(context([]), recognized("mixed", "Force", 1)).summary
    assert "wrote “Force” and drew 1 freehand stroke " in mixed
    shapes = interp.interpret(context([], native=None), recognized("shapes_only", None)).summary
    assert "drew shapes with no text" in shapes
    empty = interp.interpret(context([], native=None), recognized("empty", None)).summary
    assert "no longer on the board" in empty


def test_diagram_summary_leads_with_the_sentence_and_cites_it():
    ctx = context([], diagram=["help → needed"], native="help / needed")
    out = EvidenceInterpreter().interpret(ctx, recognized("native_text", "help / needed"))
    assert "drew a diagram: help → needed." in out.summary
    assert "diagram:0" in out.cited_evidence_ids
    assert out.category == "diagram"


def test_no_speech_is_stated_not_invented():
    out = EvidenceInterpreter().interpret(context([]), recognized())
    assert "No speech was found near this moment." in out.summary
    assert not any(c.startswith("speech:") for c in out.cited_evidence_ids)


def test_everything_quoted_exists_in_the_evidence():
    ctx = context(
        [
            speech_ev(1, 98, 104, "latency is how long one request waits"),
            speech_ev(2, 106, 110, "and throughput is requests per second", "after"),
        ],
        diagram=["Latency → Throughput"],
    )
    out = EvidenceInterpreter().interpret(ctx, recognized("native_text", "Latency"))
    allowed = " ".join(
        [SLIDE_TEXT, ctx.native_text or "", *ctx.diagram_sentences, *(s.text for s in ctx.speech)]
    )
    for quoted in re.findall(r"“([^”]+)”", out.summary):
        assert quoted.rstrip("…") in allowed
    assert set(out.cited_evidence_ids) <= set(ctx.evidence_ids())


# ------------------------------------------------------------------ whole slides


def slide_events():
    return [
        {**text("lat", 100.0, "Latency"), "pageId": S},
        {**text("thr", 400.0, "Throughput", y=300.0), "pageId": S},
    ]


def timeline_and_groups():
    tl = build_timeline(slide_events(), lecture_id=L, slide_id=S)
    return tl, group_timeline(tl)


def test_explain_slide_is_real_inputs_only_and_deterministic():
    tl, groups = timeline_and_groups()
    speech = [
        SpeechSegment(id=0, start=98, end=104, text="latency is the delay of a request"),
        SpeechSegment(id=1, start=398, end=404, text="throughput is requests per second"),
    ]
    a = explain_slide(tl, groups, speech, SLIDE_TEXT)
    b = explain_slide(tl, groups, speech, SLIDE_TEXT)
    assert [x.model_dump() for x in a] == [x.model_dump() for x in b]
    by_text = {x.context.native_text: x for x in a}
    assert "latency is the delay" in by_text["Latency"].interpretation.summary
    assert "throughput is requests per second" in by_text["Throughput"].interpretation.summary
    for x in a:
        assert x.interpretation.method == "evidence-local-v1"
        assert x.interpretation.difficulty is None
        assert x.interpretation.comprehension == []
        assert x.context.concept_candidates == []


def test_explain_slide_without_speech_or_slide_text():
    tl, groups = timeline_and_groups()
    out = explain_slide(tl, groups, None, None)
    assert len(out) == len(groups)
    assert all("No speech was found" in x.interpretation.summary for x in out)
    assert explain_slide(tl, [], None, None) == []


# ------------------------------------------------------------------ API and report


@pytest.fixture
def client(tmp_path):
    repo = AnnotationRepository(tmp_path / "c4" / "annotations.db")
    app = FastAPI()
    app.include_router(COMPONENT.router, prefix=BASE)
    app.dependency_overrides[get_repository] = lambda: repo
    return TestClient(app)


def load(client, with_speech=True):
    body = {"lecture_id": L, "slide_id": S, "events": slide_events()}
    assert client.post(f"{BASE}/events/import", json=body).status_code == 200
    client.put(f"{BASE}/lectures/{L}/slides/{S}", json={"number": 1, "text": SLIDE_TEXT})
    if with_speech:
        segs = [
            {"id": 0, "start": 98, "end": 104, "text": "latency is the delay of a request"},
            {"id": 1, "start": 398, "end": 404, "text": "throughput is requests per second"},
        ]
        client.put(f"{BASE}/lectures/{L}/speech", json={"segments": segs, "offset_seconds": 0})


def test_report_has_what_it_means_with_evidence(client):
    load(client)
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "<h3>What it means</h3>" in html
    assert "(not AI)" not in html
    assert "latency is the delay of a request" in html
    assert "<summary>Evidence</summary>" in html
    assert "Said at 01:38" in html


def test_report_works_without_speech_or_slide_info(client):
    body = {"lecture_id": L, "slide_id": S, "events": slide_events()}
    client.post(f"{BASE}/events/import", json=body)
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "No speech was found near this moment." in html
    assert "<h3>What it means</h3>" in html


def test_report_escapes_hostile_text_in_explanations(client):
    evil = "<script>alert(1)</script>"
    events = [{**text("e", 100.0, evil), "pageId": S}]
    client.post(f"{BASE}/events/import", json={"lecture_id": L, "slide_id": S, "events": events})
    client.put(
        f"{BASE}/lectures/{L}/speech",
        json={"segments": [{"id": 0, "start": 99, "end": 101, "text": evil}], "offset_seconds": 0},
    )
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "<script>alert" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_explain_endpoint_stores_and_catchup_uses_real_data(client):
    load(client)
    resp = client.post(f"{BASE}/lectures/{L}/explain")
    assert resp.status_code == 200
    body = resp.json()
    assert body["method"] == "evidence-local-v1"
    assert (body["slide_count"], body["group_count"]) == (1, 2)
    assert (body["used_speech"], body["used_slide_text"]) == (True, True)
    items = client.get(f"{BASE}/lectures/{L}/catchup").json()["items"]
    assert len(items) == 2
    assert all("mock" not in i["summary"].lower() for i in items)
    assert items[0]["start_seconds"] <= items[1]["start_seconds"]
    assert client.post(f"{BASE}/lectures/nope/explain").status_code == 404


def test_stored_ai_style_explanation_wins_over_the_live_one(client, tmp_path):
    repo = AnnotationRepository(tmp_path / "c4" / "annotations.db")
    client.app.dependency_overrides[get_repository] = lambda: repo
    load(client, with_speech=False)
    tl = repo.timeline(L, S)
    groups = group_timeline(tl)
    repo.replace_groups(L, S, groups, GroupingConfig())
    live = explain_slide(tl, groups, None, None)
    fake = live[0].model_copy(deep=True)
    fake.interpretation = AnnotationInterpretation(
        group_id=fake.context.group_id,
        category="x",
        summary="A model-written sentence.",
        cited_evidence_ids=[f"annotation:{fake.context.group_id}"],
        method="llm:test-model",
        confidence=0.5,
    )
    repo.save_analyses(L, [GroupAnalysis(**fake.model_dump())])
    html = client.get(f"{BASE}/lectures/{L}/report").text
    assert "A model-written sentence." in html
    assert "AI-generated" not in html
    assert "(not AI)" not in html


def test_weak_one_word_matches_are_dropped_when_a_better_sentence_exists():
    ctx = context(
        [
            speech_ev(0, 60, 64, "user feels this delay", "before"),
            speech_ev(1, 106, 110, "latency means the delay of one request", "after"),
        ],
        native="latency delay request",
    )
    chosen = rank_speech(ctx, terms("latency delay request"))
    assert [s.evidence_id for s in chosen] == ["speech:1"]
    # two similarly relevant sentences are both kept
    both = context(
        [
            speech_ev(0, 99, 101, "latency is the delay"),
            speech_ev(1, 106, 110, "the delay of one request", "after"),
        ]
    )
    assert len(rank_speech(both, terms("latency delay request"))) == 2
