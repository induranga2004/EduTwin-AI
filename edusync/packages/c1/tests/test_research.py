import csv

import pytest

from edusync_c1.research import train_model
from edusync_c1.research.dataset import (
    COVERAGE_LABELS,
    REQUIRED_COLUMNS,
    load_dataset,
    save_annotation,
    validate_rows,
)


def _row(**overrides):
    row = {
        "lecture_id": "L1",
        "segment_id": "0",
        "timestamp_seconds": "1.5",
        "concept": "Gradient Descent",
        "transcript_text": "we discuss gradient descent",
        "semantic_similarity": "0.8",
        "prerequisite_coverage": "0.5",
        "previous_evidence_count": "0",
        "label": "introduced",
    }
    row.update(overrides)
    return row


def test_validate_rows_coerces_numbers():
    (clean,) = validate_rows([_row()])
    assert clean["semantic_similarity"] == 0.8
    assert clean["previous_evidence_count"] == 0.0


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "no examples"),
        ([{"lecture_id": "L"}], "Missing required columns"),
        ([_row(concept="")], "empty values"),
        ([_row(timestamp_seconds="abc")], "numeric"),
        ([_row(timestamp_seconds="-1")], "negative"),
        ([_row(previous_evidence_count="-1")], "negative"),
        ([_row(semantic_similarity="1.5")], "between 0 and 1"),
        ([_row(prerequisite_coverage="-0.1")], "between 0 and 1"),
        ([_row(label="mastered")], "Invalid coverage labels"),
    ],
)
def test_validate_rows_rejects(rows, message):
    with pytest.raises(ValueError, match=message):
        validate_rows(rows)


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "ann" / "ex.csv"
    save_annotation(path, _row())
    save_annotation(path, _row(segment_id="1", label="adequately_covered"))
    with path.open(encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 3  # header written once
    rows = load_dataset(path)
    assert [r["label"] for r in rows] == ["introduced", "adequately_covered"]
    assert set(REQUIRED_COLUMNS) <= set(rows[0])
    assert "introduced" in COVERAGE_LABELS


def test_save_annotation_validates(tmp_path):
    with pytest.raises(ValueError, match="missing fields"):
        save_annotation(tmp_path / "x.csv", {"lecture_id": "L"})
    with pytest.raises(ValueError, match="Invalid coverage"):
        save_annotation(tmp_path / "x.csv", _row(label="nope"))
    assert not (tmp_path / "x.csv").exists()


def test_load_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "none.csv")


def _many(n=24, lectures=4, labels=("introduced", "adequately_covered")):
    return validate_rows(
        [
            _row(lecture_id=f"L{i % lectures}", segment_id=str(i), label=labels[i % len(labels)])
            for i in range(n)
        ]
    )


def test_check_trainable():
    train_model.check_trainable(_many())
    with pytest.raises(ValueError, match="Only 3 examples"):
        train_model.check_trainable(_many(3))
    with pytest.raises(ValueError, match="distinct lecture"):
        train_model.check_trainable(_many(lectures=2))
    with pytest.raises(ValueError, match="two different labels"):
        train_model.check_trainable(_many(labels=("introduced",)))


def test_main_reports_errors_and_success(tmp_path, monkeypatch, capsys):
    def fail(*args):
        raise FileNotFoundError("no dataset")

    monkeypatch.setattr(train_model, "train", fail)
    assert train_model.main(["--dataset", str(tmp_path / "x.csv")]) == 1
    assert "no dataset" in capsys.readouterr().out

    monkeypatch.setattr(train_model, "train", lambda *a: {"accuracy": 0.5, "macro_f1": 0.4})
    assert train_model.main(["--model-dir", str(tmp_path), "--report-dir", str(tmp_path)]) == 0


def test_train_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    pytest.importorskip("joblib")
    path = tmp_path / "d.csv"
    for row in _many(30, 5):
        save_annotation(path, row | {"transcript_text": f"text {row['label']} {row['segment_id']}"})
    metrics = train_model.train(path, tmp_path / "m", tmp_path / "r")
    assert 0 <= metrics["accuracy"] <= 1
    assert (tmp_path / "m" / "coverage_classifier.joblib").is_file()
