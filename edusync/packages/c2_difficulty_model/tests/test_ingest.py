"""Real-deck ingestion, tested on small synthetic decks only."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import edusync_c2
from edusync_c2.difficulty.synthetic import build_pptx, make_deck_spec
from edusync_c2_model import ingest
from edusync_c2_model.examples import ConceptFeatures, FeatureRow
from edusync_c2_model.extract import extract_all
from edusync_c2_model.ingest import (
    REVIEW_STATUS,
    DeckResult,
    IngestError,
    ensure_graphs,
    feature_stats,
    ingest_decks,
    list_decks,
    load_draft_module,
    load_reference_features,
    render_summary,
)
from edusync_c2_model.manifest import read_manifest

PKG = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_dir(tmp_path: Path) -> Path:
    """A data/real-like folder with two synthetic decks, a note file and a broken deck."""
    decks = tmp_path / "real" / "decks"
    decks.mkdir(parents=True)
    for seed, name in ((3, "L01"), (4, "L02")):
        spec = make_deck_spec(seed, n_concepts=5, lecture_id=name)
        (decks / f"{name}.pptx").write_bytes(build_pptx(spec))
    (decks / "notes.txt").write_text("not a deck", encoding="utf-8")
    return tmp_path / "real"


def run(real: Path, **kw):
    return ingest_decks(
        real / "decks",
        real / "graphs",
        real / "cache",
        real / "manifest.csv",
        course="ISSA",
        **kw,
    )


def test_ingest_builds_graphs_features_and_manifest(real_dir):
    results = run(real_dir)
    assert [r.lecture_id for r in results] == ["L01", "L02"]
    assert all(r.error is None and r.rows and not r.cache_hit for r in results)
    for r in results:
        assert r.slides > 0
        assert r.nodes > 0
        assert r.draft_created
        assert r.reviewed_created
        draft = json.loads((real_dir / "graphs" / f"{r.lecture_id}.draft.json").read_text("utf-8"))
        reviewed = json.loads(
            (real_dir / "graphs" / f"{r.lecture_id}.graph.json").read_text("utf-8")
        )
        assert draft["curated"] is False
        assert reviewed["curated"] is True
        assert reviewed["review_status"] == REVIEW_STATUS
        assert reviewed["graph"] == draft["graph"]
        assert {row.graph_kind for row in r.rows} == {"curated"}
        assert all(row.course_id == "ISSA" for row in r.rows)
        assert any(row.slide_indexes for row in r.rows)
        # group B needs prerequisite edges, and the synthetic decks have no mentions
        assert r.edges == 0
        assert all(s == "unavailable" for row in r.rows for s in row.statuses[8:15])
    entries = read_manifest(real_dir / "manifest.csv")
    assert [(e.lecture_id, e.course_id) for e in entries] == [("L01", "ISSA"), ("L02", "ISSA")]
    assert all(e.graph.name.endswith(".graph.json") for e in entries)

    again = run(real_dir)
    assert all(r.cache_hit and not r.draft_created and not r.reviewed_created for r in again)
    assert [r.rows for r in again] == [r.rows for r in results]


def test_reviewed_copy_is_never_overwritten(real_dir):
    run(real_dir)
    reviewed = real_dir / "graphs" / "L01.graph.json"
    payload = json.loads(reviewed.read_text("utf-8"))
    payload["review_status"] = "reviewed by the team"
    reviewed.write_text(json.dumps(payload), encoding="utf-8")
    results = run(real_dir)
    assert json.loads(reviewed.read_text("utf-8"))["review_status"] == "reviewed by the team"
    assert results[0].error is None
    assert not results[0].cache_hit
    assert results[1].cache_hit
    # a missing reviewed copy is recreated from the existing draft
    reviewed.unlink()
    path, draft_created, reviewed_created = ensure_graphs(
        real_dir / "decks" / "L01.pptx", real_dir / "graphs"
    )
    assert (path, draft_created, reviewed_created) == (reviewed, False, True)


def test_refresh_auto_rewrites_only_unreviewed_graphs(real_dir):
    run(real_dir)
    graphs = real_dir / "graphs"
    fresh = (graphs / "L01.graph.json").read_text("utf-8")
    fresh_draft = (graphs / "L01.draft.json").read_text("utf-8")
    stale = json.loads(fresh)
    stale["page_count"] = 999
    (graphs / "L01.graph.json").write_text(json.dumps(stale), encoding="utf-8")
    (graphs / "L01.draft.json").write_text("{}", encoding="utf-8")
    edited = json.loads((graphs / "L02.graph.json").read_text("utf-8"))
    edited["review_status"] = "reviewed by the team"
    edited["page_count"] = 999
    (graphs / "L02.graph.json").write_text(json.dumps(edited), encoding="utf-8")

    plain = {r.lecture_id: r for r in run(real_dir)}  # no flag: nothing is overwritten
    assert (plain["L01"].refreshed, plain["L01"].refresh_skipped) == (False, False)
    assert json.loads((graphs / "L01.graph.json").read_text("utf-8")) == stale

    results = {r.lecture_id: r for r in run(real_dir, refresh_auto=True)}
    assert (graphs / "L01.graph.json").read_text("utf-8") == fresh
    assert (graphs / "L01.draft.json").read_text("utf-8") == fresh_draft
    assert (results["L01"].refreshed, results["L01"].refresh_skipped) == (True, False)
    assert (results["L01"].reviewed_created, results["L01"].draft_created) == (True, True)
    assert json.loads((graphs / "L02.graph.json").read_text("utf-8")) == edited
    assert (results["L02"].refreshed, results["L02"].refresh_skipped) == (False, True)
    assert results["L02"].error is None


def test_refresh_auto_treats_unreadable_or_odd_graphs_as_edited(real_dir):
    run(real_dir)
    graphs = real_dir / "graphs"
    (graphs / "L01.graph.json").write_text("{not json", encoding="utf-8")
    assert not ingest._is_auto_draft(graphs / "L01.graph.json")
    (graphs / "L02.graph.json").write_text("[]", encoding="utf-8")
    assert not ingest._is_auto_draft(graphs / "L02.graph.json")
    (graphs / "L02.graph.json").unlink()
    results = {r.lecture_id: r for r in run(real_dir, refresh_auto=True)}
    assert results["L01"].refresh_skipped
    assert (graphs / "L01.graph.json").read_text("utf-8") == "{not json"


def test_mention_edges_make_group_b_computed(tmp_path):
    from io import BytesIO

    from pptx import Presentation

    prs = Presentation()
    for title, body in [("Alpha", ""), ("Bravo", "alpha first"), ("Charlie", "bravo and alpha")]:
        slide = prs.slides.add_slide(prs.slide_layouts[5])
        slide.shapes.title.text = title
        if body:
            slide.shapes.add_textbox(0, 0, 100, 100).text_frame.text = body
    buf = BytesIO()
    prs.save(buf)
    decks = tmp_path / "real" / "decks"
    decks.mkdir(parents=True)
    (decks / "M01.pptx").write_bytes(buf.getvalue())
    (result,) = run(tmp_path / "real")
    assert (result.error, result.edges) == (None, 3)
    assert any(s == "computed" for row in result.rows for s in row.statuses[8:15])


def test_failures_are_reported_by_type_and_skipped(real_dir):
    (real_dir / "decks" / "L03.pptx").write_bytes(b"not a presentation")
    (real_dir / "decks" / "bad name.pptx").write_bytes(b"x")
    results = {r.lecture_id: r for r in run(real_dir)}
    assert results["L03"].error == "MaterialsError"
    assert results["bad name"].error == "IngestError"
    assert results["L01"].error is None
    assert [e.lecture_id for e in read_manifest(real_dir / "manifest.csv")] == ["L01", "L02"]


def test_no_manifest_when_nothing_ingests(tmp_path):
    decks = tmp_path / "decks"
    decks.mkdir()
    (decks / "L01.pptx").write_bytes(b"broken")
    results = ingest_decks(
        decks, tmp_path / "g", tmp_path / "c", tmp_path / "manifest.csv", course="C"
    )
    assert results[0].error == "MaterialsError"
    assert not (tmp_path / "manifest.csv").exists()


def test_draft_that_breaks_the_contract_is_an_error(real_dir, monkeypatch):
    monkeypatch.setattr(ingest, "check_c1_graph", lambda *_: SimpleNamespace(ok=False))
    results = run(real_dir)
    assert {r.error for r in results} == {"IngestError"}
    assert not (real_dir / "graphs" / "L01.draft.json").exists()


def test_draft_tool_lookup(tmp_path, monkeypatch):
    assert callable(load_draft_module().build_draft)
    fake = tmp_path / "a" / "b" / "c" / "edusync_c2" / "__init__.py"
    monkeypatch.setattr(edusync_c2, "__file__", str(fake))
    with pytest.raises(IngestError, match="not found"):
        load_draft_module()


def test_draft_tool_without_a_loader(monkeypatch):
    monkeypatch.setattr(importlib.util, "spec_from_file_location", lambda *a, **k: None)
    with pytest.raises(IngestError, match="not found"):
        load_draft_module()


def test_list_decks_filters_and_sorts(real_dir):
    assert [p.name for p in list_decks(real_dir / "decks")] == ["L01.pptx", "L02.pptx"]


def test_load_reference_features_is_sorted_and_label_free(real_dir):
    run(real_dir)
    from_path = load_reference_features(real_dir / "cache", real_dir / "manifest.csv")
    entries = read_manifest(real_dir / "manifest.csv")
    from_entries = load_reference_features(real_dir / "cache", list(reversed(entries)))
    assert from_path == from_entries
    assert from_path == sorted(from_path, key=lambda r: r.key)
    assert {r.lecture_id for r in from_path} == {"L01", "L02"}
    assert all(isinstance(r, ConceptFeatures) and not hasattr(r, "label") for r in from_path)
    rows, stats = extract_all(entries, real_dir / "cache")
    assert stats.misses == 0
    assert len(rows) == len(from_path)


def row(statuses, values):
    return FeatureRow(
        "L1", "C", "c", "c", "t", tuple(values), tuple(0 for _ in values), tuple(statuses)
    )


def test_feature_stats_flags_and_ranges():
    n = len(ingest.FEATURE_SPECS)
    s1 = ["computed"] * n
    s2 = ["computed"] * n
    s1[1], s2[1] = "unavailable", "unavailable"  # all unavailable
    s1[2], s2[2] = "mocked", "unavailable"  # never computed
    v1 = [float(i) for i in range(n)]
    v2 = [float(i) for i in range(n)]
    v2[4] = v1[4] + 2.0  # varying -> no flag, median of two
    stats = feature_stats([row(s1, v1), row(s2, v2)])
    assert stats[0].flag == "constant"
    assert stats[0].counts["computed"] == 2
    assert stats[1].flag == "all unavailable"
    assert stats[1].low is None
    assert stats[2].flag == "never computed"
    assert stats[4].flag == ""
    assert (stats[4].low, stats[4].median, stats[4].high) == (
        4.0,
        5.0,
        6.0,
    )


def test_render_summary_has_counts_and_no_text(real_dir):
    results = run(real_dir)
    results.append(DeckResult(lecture_id="L99", error="MaterialsError"))
    text = render_summary(results, course="ISSA")
    assert "2 ingested, 1 failed" in text
    assert "| L99 | 0 | 0 | 0 | 0 | miss | 0.00 | MaterialsError |" in text
    assert "UNREVIEWED" in text
    assert "semantic" in text
    for result in results:
        for r in result.rows:
            assert r.text not in text
    assert text.count("\n| ") > 35


def load_script():
    path = PKG / "scripts" / "ingest_real_decks.py"
    spec = importlib.util.spec_from_file_location("ingest_real_decks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_main(real_dir, capsys, tmp_path):
    script = load_script()
    args = [
        "--decks", str(real_dir / "decks"),
        "--graphs", str(real_dir / "graphs"),
        "--cache", str(real_dir / "cache"),
        "--summary", str(tmp_path / "runs" / "s.md"),
    ]  # fmt: skip
    assert script.main(args) == 0
    assert "2 of 2 decks ingested" in capsys.readouterr().out
    assert (tmp_path / "runs" / "s.md").is_file()
    assert (real_dir / "manifest.csv").is_file()

    edited = json.loads((real_dir / "graphs" / "L02.graph.json").read_text("utf-8"))
    edited["review_status"] = "reviewed"
    (real_dir / "graphs" / "L02.graph.json").write_text(json.dumps(edited), encoding="utf-8")
    assert script.main([*args, "--refresh-auto"]) == 0
    out = capsys.readouterr().out
    assert "1 refreshed, 1 left alone" in out
    assert "L02: graph was edited" in out

    (real_dir / "decks" / "L03.pptx").write_bytes(b"broken")
    assert script.main(args) == 1
    assert "L03: MaterialsError" in capsys.readouterr().err

    assert script.main([*args[:1], str(real_dir / "nope"), *args[2:]]) == 1
    empty = real_dir / "empty"
    empty.mkdir()
    assert script.main(["--decks", str(empty), *args[2:]]) == 1
