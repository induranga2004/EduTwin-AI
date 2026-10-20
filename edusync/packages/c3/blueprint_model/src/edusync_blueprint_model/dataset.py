"""CSV loading, label mapping and stratified splitting."""

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_PER_CLASS = 3


@dataclass(frozen=True)
class Example:
    text: str
    label: str


class LabelMap:
    """Stable mapping between label names (sorted) and integer ids."""

    def __init__(self, labels: list[str]) -> None:
        self.labels = sorted(set(labels))
        self._ids = {label: i for i, label in enumerate(self.labels)}

    def __len__(self) -> int:
        return len(self.labels)

    def id_of(self, label: str) -> int:
        try:
            return self._ids[label]
        except KeyError:
            raise KeyError(f"Unknown label {label!r}; known labels: {self.labels}") from None

    def label_of(self, index: int) -> str:
        return self.labels[index]

    def to_json(self) -> str:
        return json.dumps({"labels": self.labels}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "LabelMap":
        return cls(json.loads(text)["labels"])


@dataclass(frozen=True)
class Splits:
    train: list[Example]
    val: list[Example]
    test: list[Example]


def load_csv(path: Path, text_column: str, label_column: str) -> list[Example]:
    """Read and validate a labelled CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = [c for c in (text_column, label_column) if c not in fields]
        if missing:
            raise ValueError(f"Column(s) {missing} not found in {path.name}; found {fields}")
        examples = []
        for line, row in enumerate(reader, start=2):
            text = (row[text_column] or "").strip()
            label = (row[label_column] or "").strip()
            if not text or not label:
                raise ValueError(f"{path.name} line {line}: blank text or label")
            examples.append(Example(text, label))
    counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        counts[ex.label] += 1
    if len(counts) < 2:
        raise ValueError(f"Need at least 2 classes, found {len(counts)} in {path.name}")
    small = {k: v for k, v in counts.items() if v < MIN_PER_CLASS}
    if small:
        raise ValueError(f"Each class needs >= {MIN_PER_CLASS} rows; too few: {small}")
    return examples


def split_dataset(
    examples: list[Example], val_fraction: float, test_fraction: float, seed: int
) -> Splits:
    """Deterministic stratified split with at least one val and test example per class."""
    by_label: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_label[ex.label].append(ex)
    rng = random.Random(seed)
    train: list[Example] = []
    val: list[Example] = []
    test: list[Example] = []
    for label in sorted(by_label):
        items = list(by_label[label])
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, round(n * val_fraction)) if val_fraction > 0 else 0
        n_test = max(1, round(n * test_fraction)) if test_fraction > 0 else 0
        if n - n_val - n_test < 1:
            raise ValueError(
                f"Class {label!r} has {n} rows, too few for val/test fractions "
                f"({val_fraction}/{test_fraction}); add data or lower the fractions"
            )
        val.extend(items[:n_val])
        test.extend(items[n_val : n_val + n_test])
        train.extend(items[n_val + n_test :])
    rng.shuffle(train)
    return Splits(train, val, test)


def load_splits(config: Any) -> tuple[LabelMap, Splits]:
    """Load the configured CSV and split it; returns the label map and the three splits."""
    data = config.data
    examples = load_csv(data.path, data.text_column, data.label_column)
    labels = LabelMap([e.label for e in examples])
    return labels, split_dataset(examples, data.val_fraction, data.test_fraction, data.seed)
