from pathlib import Path

import pytest

from edusync_blueprint_model.config import load_config
from edusync_blueprint_model.dataset import (
    Example,
    LabelMap,
    load_csv,
    load_splits,
    split_dataset,
)


def make_csv(path: Path, counts: dict[str, int]) -> Path:
    rows = ["text,label"]
    for label, n in counts.items():
        rows += [f"t{i} {label},{label}" for i in range(n)]
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


def test_label_map_roundtrip() -> None:
    lm = LabelMap(["b", "a", "b", "c"])
    assert lm.labels == ["a", "b", "c"]
    assert len(lm) == 3
    assert lm.id_of("b") == 1
    assert lm.label_of(2) == "c"
    assert LabelMap.from_json(lm.to_json()).labels == lm.labels
    with pytest.raises(KeyError, match="Unknown label"):
        lm.id_of("z")


def test_load_csv_ok(tmp_path: Path) -> None:
    path = make_csv(tmp_path / "d.csv", {"a": 3, "b": 4})
    examples = load_csv(path, "text", "label")
    assert len(examples) == 7
    assert examples[0] == Example("t0 a", "a")


def test_load_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "missing.csv", "text", "label")
    path = make_csv(tmp_path / "d.csv", {"a": 3, "b": 3})
    with pytest.raises(ValueError, match="not found"):
        load_csv(path, "nope", "label")
    blank = tmp_path / "blank.csv"
    blank.write_text("text,label\n  ,a\n")
    with pytest.raises(ValueError, match="blank"):
        load_csv(blank, "text", "label")
    with pytest.raises(ValueError, match="at least 2 classes"):
        load_csv(make_csv(tmp_path / "one.csv", {"a": 5}), "text", "label")
    with pytest.raises(ValueError, match="too few"):
        load_csv(make_csv(tmp_path / "few.csv", {"a": 5, "b": 2}), "text", "label")


def test_split_is_stratified_and_deterministic(tmp_path: Path) -> None:
    examples = load_csv(make_csv(tmp_path / "d.csv", {"a": 10, "b": 10, "c": 10}), "text", "label")
    s1 = split_dataset(examples, 0.2, 0.2, seed=7)
    s2 = split_dataset(examples, 0.2, 0.2, seed=7)
    assert s1 == s2
    assert split_dataset(examples, 0.2, 0.2, seed=8) != s1
    assert len(s1.train) == 18
    assert len(s1.val) == 6
    assert len(s1.test) == 6
    for part in (s1.val, s1.test):
        assert {e.label for e in part} == {"a", "b", "c"}
    assert len({e.text for e in s1.train + s1.val + s1.test}) == 30


def test_split_minimum_one_per_class(tmp_path: Path) -> None:
    examples = load_csv(make_csv(tmp_path / "d.csv", {"a": 3, "b": 3}), "text", "label")
    s = split_dataset(examples, 0.01, 0.01, seed=0)
    assert len(s.val) == 2
    assert len(s.test) == 2
    assert len(s.train) == 2


def test_split_without_val_or_test(tmp_path: Path) -> None:
    examples = load_csv(make_csv(tmp_path / "d.csv", {"a": 4, "b": 4}), "text", "label")
    s = split_dataset(examples, 0.0, 0.0, seed=0)
    assert not s.val
    assert not s.test
    assert len(s.train) == 8


def test_split_tiny_class_raises(tmp_path: Path) -> None:
    examples = load_csv(make_csv(tmp_path / "d.csv", {"a": 3, "b": 3}), "text", "label")
    with pytest.raises(ValueError, match="too few"):
        split_dataset(examples, 0.6, 0.3, seed=0)


def test_load_splits(config_file: Path) -> None:
    labels, splits = load_splits(load_config(config_file))
    assert labels.labels == ["a", "b", "c"]
    assert len(splits.test) == 6
