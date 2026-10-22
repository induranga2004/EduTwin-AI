"""Manifest validation and the Example record."""

import json
from pathlib import Path

import pytest

from edusync_c2.difficulty.registry import FEATURE_NAMES
from edusync_c2_model.examples import (
    N_FEATURES,
    Example,
    FeatureRow,
    read_examples,
    write_examples,
)
from edusync_c2_model.ids import ID_PATTERN, is_valid_id
from edusync_c2_model.manifest import (
    ManifestEntry,
    ManifestError,
    read_manifest,
    write_manifest,
)


def test_id_rule_matches_the_export_label_sheet_script(c2_script):
    script = c2_script("export_label_sheet")
    assert ID_PATTERN.pattern == script.LECTURE_ID_PATTERN.pattern
    for value in ("L001", "a_b-c", "x" * 32, "", "x" * 33, "a b", "../x", "-a"):
        assert is_valid_id(value) == bool(script.LECTURE_ID_PATTERN.match(value))


def write_manifest_text(folder: Path, body: str) -> Path:
    path = folder / "manifest.csv"
    path.write_text(body, encoding="utf-8", newline="")
    return path


def test_read_manifest_resolves_relative_and_absolute_paths(tmp_path):
    (tmp_path / "a.pptx").write_bytes(b"x")
    (tmp_path / "g.json").write_text("{}")
    absolute = tmp_path / "b.pptx"
    absolute.write_bytes(b"x")
    path = write_manifest_text(
        tmp_path,
        f"lecture_id,course_id,file,graph\nL1,C1,a.pptx,g.json\n\nL2,C2,{absolute},\n",
    )
    entries = read_manifest(path)
    assert entries == [
        ManifestEntry("L1", "C1", tmp_path / "a.pptx", tmp_path / "g.json"),
        ManifestEntry("L2", "C2", absolute, None),
    ]


def test_graph_column_is_optional(tmp_path):
    (tmp_path / "a.pptx").write_bytes(b"x")
    path = write_manifest_text(tmp_path, "lecture_id,course_id,file\nL1,C1,a.pptx\n")
    assert read_manifest(path)[0].graph is None


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("lecture_id,course_id\nL1,C1\n", "missing column"),
        ("lecture_id,course_id,file\n", "no lectures"),
        ("lecture_id,course_id,file\nbad id,C1,a.pptx\n", "lecture_id 'bad id'"),
        ("lecture_id,course_id,file\nL1,../C1,a.pptx\n", "course_id"),
        ("lecture_id,course_id,file\nL1,C1,a.pptx\nL1,C2,a.pptx\n", "duplicate lecture_id"),
        ("lecture_id,course_id,file\nL1,C1,nope.pptx\n", "lecture file does not exist"),
        ("lecture_id,course_id,file,graph\nL1,C1,a.pptx,nope.json\n", "graph file does not exist"),
    ],
)
def test_invalid_manifests_raise(tmp_path, body, message):
    (tmp_path / "a.pptx").write_bytes(b"x")
    with pytest.raises(ManifestError, match=message):
        read_manifest(write_manifest_text(tmp_path, body))


def test_unreadable_manifest_raises(tmp_path):
    with pytest.raises(ManifestError, match="cannot read manifest"):
        read_manifest(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ManifestError, match="cannot read manifest"):
        read_manifest(bad)


def test_write_manifest_round_trips(tmp_path):
    (tmp_path / "lectures").mkdir()
    (tmp_path / "lectures" / "a.pptx").write_bytes(b"x")
    (tmp_path / "lectures" / "a.json").write_text("{}")
    entries = [
        ManifestEntry(
            "L1", "C1", tmp_path / "lectures" / "a.pptx", tmp_path / "lectures" / "a.json"
        ),
        ManifestEntry("L2", "C1", tmp_path / "lectures" / "a.pptx", None),
    ]
    path = tmp_path / "manifest.csv"
    write_manifest(path, entries)
    assert read_manifest(path) == entries


# ------------------------------------------------------------------------------ examples


def vectors(n=N_FEATURES):
    return tuple(float(i) / 7 for i in range(n)), (0,) * n, ("computed",) * n


def make_row(**overrides):
    values, missing, statuses = vectors()
    base = {
        "lecture_id": "L1",
        "course_id": "C1",
        "concept_id": "a",
        "concept_label": "A",
        "text": "A. text\nmore",
        "values": values,
        "missing": missing,
        "statuses": statuses,
    }
    return FeatureRow(**{**base, **overrides})


def test_there_are_35_features():
    assert N_FEATURES == len(FEATURE_NAMES) == 35


def test_feature_row_round_trips_and_is_deterministic():
    row = make_row()
    again = FeatureRow.from_json(row.to_json())
    assert again == row
    assert row.to_json() == make_row().to_json()
    assert json.loads(row.to_json())["concept_id"] == "a"
    assert row.key == ("L1", "a")
    assert " " not in row.to_json().split('"text"')[0]  # compact separators


def test_example_round_trips_and_json_is_sorted():
    example = Example.from_row(make_row(), "High", 3, True)
    assert (example.label_id, example.n_raters, example.synthetic) == (2, 3, True)
    assert Example.from_json(example.to_json()) == example
    keys = list(json.loads(example.to_json()))
    assert keys == sorted(keys)
    assert example.to_json() == Example.from_row(make_row(), "High", 3, True).to_json()


def test_vectors_must_have_35_entries():
    values, missing, statuses = vectors(34)
    with pytest.raises(ValueError, match="35"):
        make_row(values=values, missing=missing, statuses=statuses)
    with pytest.raises(ValueError, match="35"):
        Example("L1", "C1", "a", "A", "t", values, missing, statuses, "Low", 0, 1, False)


def test_label_and_label_id_must_agree():
    values, missing, statuses = vectors()
    with pytest.raises(ValueError, match="disagree"):
        Example("L1", "C1", "a", "A", "t", values, missing, statuses, "Low", 2, 1, False)
    with pytest.raises(ValueError, match="disagree"):
        Example("L1", "C1", "a", "A", "t", values, missing, statuses, "Easy", 0, 1, False)


def test_write_and_read_examples(tmp_path):
    examples = [Example.from_row(make_row(concept_id=c), "Low", 1, False) for c in "ab"]
    path = tmp_path / "e.jsonl"
    assert write_examples(path, examples) == 2
    assert read_examples(path) == examples
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")
    assert read_examples(path) == examples
