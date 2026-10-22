"""The training example: one labelled concept with exactly the inputs the model sees at inference.

``FeatureRow`` is the unlabelled extraction result; ``Example`` adds the resolved label. Both
serialise to deterministic one-line JSON (sorted keys, compact separators).
"""

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from edusync_c2.difficulty.models import LABEL_ORDER
from edusync_c2.difficulty.registry import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)
LABEL_ID = {name: i for i, name in enumerate(LABEL_ORDER)}


def _check_vectors(values: tuple, missing: tuple, statuses: tuple) -> None:
    if not len(values) == len(missing) == len(statuses) == N_FEATURES:
        raise ValueError(f"values, missing and statuses must each have {N_FEATURES} entries")


def _dumps(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FeatureRow:
    lecture_id: str
    course_id: str
    concept_id: str
    concept_label: str
    text: str
    values: tuple[float, ...]
    missing: tuple[int, ...]
    statuses: tuple[str, ...]
    slide_indexes: tuple[int, ...] = ()
    graph_kind: str = ""  # "real", "mock", "curated" or "none": where the graph came from

    def __post_init__(self) -> None:
        _check_vectors(self.values, self.missing, self.statuses)

    @property
    def key(self) -> tuple[str, str]:
        return (self.lecture_id, self.concept_id)

    def to_json(self) -> str:
        return _dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "FeatureRow":
        data = json.loads(text)
        return cls(
            lecture_id=data["lecture_id"],
            course_id=data["course_id"],
            concept_id=data["concept_id"],
            concept_label=data["concept_label"],
            text=data["text"],
            values=tuple(float(v) for v in data["values"]),
            missing=tuple(int(m) for m in data["missing"]),
            statuses=tuple(data["statuses"]),
            slide_indexes=tuple(int(i) for i in data.get("slide_indexes", ())),
            graph_kind=str(data.get("graph_kind", "")),
        )


# The label-free record used for reference scoring: same fields, no label.
ConceptFeatures = FeatureRow


@dataclass(frozen=True)
class Example:
    lecture_id: str
    course_id: str
    concept_id: str
    concept_label: str
    text: str
    values: tuple[float, ...]
    missing: tuple[int, ...]
    statuses: tuple[str, ...]
    label: str
    label_id: int
    n_raters: int
    synthetic: bool

    def __post_init__(self) -> None:
        _check_vectors(self.values, self.missing, self.statuses)
        if self.label not in LABEL_ID or LABEL_ID[self.label] != self.label_id:
            raise ValueError(f"label {self.label!r} and label_id {self.label_id} disagree")

    @classmethod
    def from_row(cls, row: FeatureRow, label: str, n_raters: int, synthetic: bool) -> "Example":
        return cls(
            lecture_id=row.lecture_id,
            course_id=row.course_id,
            concept_id=row.concept_id,
            concept_label=row.concept_label,
            text=row.text,
            values=row.values,
            missing=row.missing,
            statuses=row.statuses,
            label=label,
            label_id=LABEL_ORDER.index(label),
            n_raters=n_raters,
            synthetic=synthetic,
        )

    def to_json(self) -> str:
        return _dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Example":
        data = json.loads(text)
        return cls(
            lecture_id=data["lecture_id"],
            course_id=data["course_id"],
            concept_id=data["concept_id"],
            concept_label=data["concept_label"],
            text=data["text"],
            values=tuple(float(v) for v in data["values"]),
            missing=tuple(int(m) for m in data["missing"]),
            statuses=tuple(data["statuses"]),
            label=data["label"],
            label_id=int(data["label_id"]),
            n_raters=int(data["n_raters"]),
            synthetic=bool(data["synthetic"]),
        )


def write_examples(path: Path, examples: Iterable[Example]) -> int:
    """Write one JSON line per example; returns the count."""
    count = 0
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.to_json() + "\n")
            count += 1
    return count


def read_examples(path: Path) -> list[Example]:
    with open(path, encoding="utf-8") as handle:
        return [Example.from_json(line) for line in handle if line.strip()]
