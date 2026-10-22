"""Read filled label CSVs and resolve one label per ``(lecture_id, concept_id)``.

Input columns follow docs/edusync/c2-label-collection.md section 9 (only ``lecture_id``,
``concept_id``, ``difficulty_label``, ``rater_id`` and ``synthetic`` are needed here; run
``validate_labels.py`` first for the full rule set). Rows with an empty ``difficulty_label`` are
unlabelled sheet rows and are skipped.

Resolution rule per concept:

1. An adjudicator row (rater ``ADJ``) wins; an ``ADJ`` row labelled ``CannotJudge`` drops it.
2. Otherwise ``CannotJudge`` rows are removed; no rows left drops the concept.
3. Otherwise the ordinal median of the raters (Low=0, Medium=1, High=2). With an even number of
   raters whose two middle labels differ the median is a tie: the concept is excluded and counted.

``n_raters`` counts the non-adjudicator raters with a usable label; ``raw_labels`` lists their
labels ordered by rater id.
"""

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

LABEL_ORDER: tuple[str, ...] = ("Low", "Medium", "High")  # equals C2's LABEL_ORDER (tested)
CANNOT_JUDGE = "CannotJudge"
ADJUDICATOR = "ADJ"
LABEL_ID = {name: i for i, name in enumerate(LABEL_ORDER)}
REQUIRED_COLUMNS = ("lecture_id", "concept_id", "difficulty_label", "rater_id", "synthetic")

Key = tuple[str, str]


class LabelError(ValueError):
    """A label file or row cannot be used."""


@dataclass(frozen=True)
class ResolvedLabel:
    lecture_id: str
    concept_id: str
    label: str
    label_id: int
    n_raters: int
    raw_labels: tuple[str, ...]
    source: str  # "adjudicated" | "median"
    synthetic: bool


@dataclass
class LabelSet:
    """Resolved labels plus the counts of everything that was not turned into a label."""

    labels: dict[Key, ResolvedLabel] = field(default_factory=dict)
    ties_excluded: int = 0
    cannot_judge_dropped: int = 0
    unlabelled_rows: int = 0

    def __len__(self) -> int:
        return len(self.labels)


def read_rows(path: Path) -> list[dict[str, str]]:
    """Stripped rows of one label CSV; blank rows are skipped."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = [h.strip() for h in (reader.fieldnames or [])]
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                raise LabelError(f"{path.name}: missing column(s): {', '.join(missing)}")
            rows = []
            for raw in reader:
                row = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k}
                if any(row.values()):
                    rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise LabelError(f"cannot read {path}: {exc}") from exc
    return rows


def _median_id(ids: list[int]) -> int | None:
    """Ordinal median of sorted ids; None for an even count with different middle labels."""
    n = len(ids)
    if n % 2:
        return ids[n // 2]
    lower, upper = ids[n // 2 - 1], ids[n // 2]
    return lower if lower == upper else None


def resolve_labels(rows: Iterable[dict[str, str]]) -> LabelSet:
    """Resolve rater rows into one label per concept (see the module docstring)."""
    result = LabelSet()
    grouped: dict[Key, dict[str, str]] = {}
    synthetic: dict[Key, bool] = {}
    for row in rows:
        label = row["difficulty_label"]
        if not label:
            result.unlabelled_rows += 1
            continue
        if label != CANNOT_JUDGE and label not in LABEL_ID:
            raise LabelError(f"unknown difficulty_label {label!r}")
        lecture, concept, rater = row["lecture_id"], row["concept_id"], row["rater_id"]
        if not lecture or not concept or not rater:
            raise LabelError("lecture_id, concept_id and rater_id must not be empty")
        if row["synthetic"] not in ("true", "false"):
            raise LabelError(f"synthetic must be true or false, got {row['synthetic']!r}")
        key = (lecture, concept)
        raters = grouped.setdefault(key, {})
        if rater in raters:
            raise LabelError(f"duplicate label for {concept!r} of {lecture!r} by rater {rater!r}")
        raters[rater] = label
        synthetic[key] = synthetic.get(key, False) or row["synthetic"] == "true"

    for key in sorted(grouped):
        raters = grouped[key]
        usable = sorted((r, lab) for r, lab in raters.items() if r != ADJUDICATOR)
        usable = [(r, lab) for r, lab in usable if lab != CANNOT_JUDGE]
        raw = tuple(lab for _, lab in usable)
        adjudicated = raters.get(ADJUDICATOR)
        if adjudicated is not None:
            if adjudicated == CANNOT_JUDGE:
                result.cannot_judge_dropped += 1
                continue
            chosen, source = LABEL_ID[adjudicated], "adjudicated"
        elif not usable:
            result.cannot_judge_dropped += 1
            continue
        else:
            median = _median_id(sorted(LABEL_ID[lab] for lab in raw))
            if median is None:
                result.ties_excluded += 1
                continue
            chosen, source = median, "median"
        result.labels[key] = ResolvedLabel(
            lecture_id=key[0],
            concept_id=key[1],
            label=LABEL_ORDER[chosen],
            label_id=chosen,
            n_raters=len(usable),
            raw_labels=raw,
            source=source,
            synthetic=synthetic[key],
        )
    return result


def read_labels(paths: Iterable[Path]) -> LabelSet:
    """Read and resolve one or more filled label CSVs together."""
    rows: list[dict[str, str]] = []
    for path in paths:
        rows.extend(read_rows(Path(path)))
    return resolve_labels(rows)
