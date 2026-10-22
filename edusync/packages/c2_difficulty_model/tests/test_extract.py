"""Feature extraction through the frozen pipeline, its cache, and the label join."""

import json

import pytest

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.schemas import C1GraphResponse
from edusync_c2_model import extract
from edusync_c2_model.extract import (
    ExtractError,
    RecordingModel,
    extract_all,
    extract_lecture,
    join_examples,
)
from edusync_c2_model.labels import LabelSet, ResolvedLabel, resolve_labels
from edusync_c2_model.manifest import ManifestEntry


def test_recording_model_captures_and_delegates(make_entry):
    entry = make_entry()
    recorder = RecordingModel()
    pipeline = DifficultyPipeline(model=recorder)
    graph = C1GraphResponse.model_validate(json.loads(entry.graph.read_text(encoding="utf-8")))
    response = pipeline.run(entry.file.read_bytes(), entry.file.name, graph)
    assert len(recorder.captured) == len(response.predictions) > 0
    assert recorder.info.name == "rule_baseline"
    # delegating means predictions are exactly the rule baseline's
    expected = recorder._inner.predict(recorder.captured)
    assert [p.difficulty for p in response.predictions] == [
        o.label if o else None for o in expected
    ]


def test_rows_match_what_the_pipeline_reports(make_entry):
    entry = make_entry()
    rows = extract_lecture(entry)
    graph = C1GraphResponse.model_validate(json.loads(entry.graph.read_text(encoding="utf-8")))
    response = DifficultyPipeline().run(entry.file.read_bytes(), entry.file.name, graph)
    assert [r.concept_id for r in rows] == [p.concept_id for p in response.predictions]
    for row, prediction in zip(rows, response.predictions, strict=True):
        assert row.concept_label == prediction.label
        assert row.statuses == tuple(f.status.value for f in prediction.features)
        assert len(row.values) == len(row.missing) == 35
        for value, missing, feature in zip(
            row.values, row.missing, prediction.features, strict=True
        ):
            assert missing == int(feature.value is None)
            assert value == (feature.value if feature.value is not None else 0.0)
        assert row.lecture_id == "L001"
        assert row.course_id == "C1"
        assert row.text.startswith(prediction.label)


def test_extraction_without_graph_uses_the_mock_graph(make_entry):
    rows = extract_lecture(make_entry(graph=False))
    assert rows
    prereq = [i for i, s in enumerate(rows[0].statuses) if s != "computed"]
    assert prereq  # group B is not computed from a mock graph


def test_invalid_inputs_raise_extract_errors(make_entry, tmp_path):
    entry = make_entry()
    bad_graph = tmp_path / "bad.json"
    bad_graph.write_text("[1]", encoding="utf-8")
    with pytest.raises(ExtractError, match="not a C1 graph"):
        extract_lecture(ManifestEntry("L9", "C1", entry.file, bad_graph))
    bad_graph.write_text("not json", encoding="utf-8")
    with pytest.raises(ExtractError, match="not a C1 graph"):
        extract_lecture(ManifestEntry("L9", "C1", entry.file, bad_graph))
    broken = tmp_path / "broken.pptx"
    broken.write_bytes(b"this is not a presentation")
    with pytest.raises(ExtractError, match="rejected"):
        extract_lecture(ManifestEntry("L9", "C1", broken, None))


def test_cache_hit_then_invalidation_on_every_input(make_entry, tmp_path, monkeypatch):
    entry = make_entry()
    cache = tmp_path / "cache"
    first, stats = extract_all([entry], cache)
    assert (stats.hits, stats.misses) == (0, 1)
    again, stats = extract_all([entry], cache)
    assert (stats.hits, stats.misses) == (1, 0)
    assert again == first

    # changed concept filter, schema version, schema hash and pipeline version each miss
    _, stats = extract_all([entry], cache, concept_filter="heuristic")
    assert stats.misses == 1
    monkeypatch.setattr(extract, "FEATURE_SCHEMA_VERSION", "9.9")
    assert extract_all([entry], cache)[1].misses == 1
    monkeypatch.undo()
    monkeypatch.setattr(extract, "feature_schema_hash", lambda: "other-hash")
    assert extract_all([entry], cache)[1].misses == 1
    monkeypatch.undo()
    monkeypatch.setattr(extract, "PIPELINE_VERSION", "9.9.9")
    assert extract_all([entry], cache)[1].misses == 1
    monkeypatch.undo()
    assert extract_all([entry], cache)[1].hits == 1

    # changed lecture bytes and changed graph bytes each miss
    entry.graph.write_text(entry.graph.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert extract_all([entry], cache)[1].misses == 1
    entry.file.write_bytes(make_entry(seed=9, lecture_id="LX").file.read_bytes())
    assert extract_all([entry], cache)[1].misses == 1
    assert extract_all([entry], cache)[1].hits == 1
    # a different course id also changes the key
    other = ManifestEntry(entry.lecture_id, "C2", entry.file, entry.graph)
    assert extract_all([other], cache)[1].misses == 1


def test_corrupt_or_foreign_cache_files_are_recomputed(make_entry, tmp_path):
    entry = make_entry()
    cache = tmp_path / "cache"
    expected, _ = extract_all([entry], cache)
    (path,) = cache.glob("*.jsonl")
    for damage in ("", "not json\n", json.dumps({"other": 1}) + "\n", "{}\n{bad row}\n"):
        path.write_text(damage, encoding="utf-8")
        rows, stats = extract_all([entry], cache)
        assert (stats.misses, rows) == (1, expected)
    path.unlink()
    assert extract_all([entry], cache)[1].misses == 1


def test_extraction_is_independent_of_manifest_order(make_entry, tmp_path):
    entries = [make_entry(seed=s, lecture_id=f"L{s}", course=f"C{s % 2}") for s in (3, 4, 5)]
    forward, _ = extract_all(entries, tmp_path / "c1")
    backward, _ = extract_all(list(reversed(entries)), tmp_path / "c2")
    assert sorted(forward, key=lambda r: r.key) == sorted(backward, key=lambda r: r.key)


def label(lecture, concept, text="Low", n=2, synthetic=False):
    return ResolvedLabel(
        lecture, concept, text, ["Low", "Medium", "High"].index(text), n, (), "median", synthetic
    )


def test_join_reports_unmatched_and_unlabelled(make_entry):
    rows = extract_lecture(make_entry())
    keep, spare = rows[0], rows[1]
    labels = LabelSet(
        labels={
            keep.key: label(keep.lecture_id, keep.concept_id, "High", 3, True),
            ("L001", "no-such-concept"): label("L001", "no-such-concept"),
        }
    )
    result = join_examples(rows, labels)
    assert [e.concept_id for e in result.examples] == [keep.concept_id]
    assert (result.examples[0].label, result.examples[0].n_raters) == ("High", 3)
    assert result.examples[0].synthetic is True
    assert result.unmatched_labels == [("L001", "no-such-concept")]
    assert spare.key in result.unlabelled
    assert result.examples[0].values == keep.values


def test_join_is_order_independent_and_rejects_duplicates(make_entry):
    rows = extract_lecture(make_entry())
    labels = resolve_labels(
        {
            "lecture_id": r.lecture_id,
            "concept_id": r.concept_id,
            "difficulty_label": "Medium",
            "rater_id": "R1",
            "synthetic": "false",
        }
        for r in rows
    )
    forward = join_examples(rows, labels)
    backward = join_examples(list(reversed(rows)), labels)
    assert forward.examples == backward.examples
    assert [e.to_json() for e in forward.examples] == [e.to_json() for e in backward.examples]
    with pytest.raises(ExtractError, match="duplicate"):
        join_examples([*rows, rows[0]], labels)
