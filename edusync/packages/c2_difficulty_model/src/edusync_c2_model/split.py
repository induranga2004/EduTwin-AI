"""Seeded group splits. Lectures never straddle train and test; courses do not either when >= 3.

All functions sort their inputs first, so the result depends only on the ids and the seed (never
on input order or ``PYTHONHASHSEED``). The sealed test split is written once with
``write_split_once`` and only read afterwards.
"""

import json
import random
from collections.abc import Iterable, Mapping
from pathlib import Path

SPLIT_VERSION = 1
MIN_COURSES_FOR_COURSE_SPLIT = 3


class SplitError(ValueError):
    """A split cannot be made or is invalid."""


class LeakageError(SplitError):
    """A group appears on both sides of a split."""


def _held_out(items: list[str], fraction: float, seed: int, label: str) -> set[str]:
    if len(items) < 2:
        raise SplitError(f"need at least 2 {label}s to split, got {len(items)}")
    order = sorted(items)
    random.Random(seed).shuffle(order)
    n_test = min(max(1, round(fraction * len(order))), len(order) - 1)
    return set(order[:n_test])


def make_split(
    lecture_courses: Mapping[str, str], *, seed: int, test_fraction: float = 0.2
) -> dict:
    """``{"unit", "seed", "train", "test", ...}``; by course when there are >= 3 courses."""
    if not 0 < test_fraction < 1:
        raise SplitError("test_fraction must be between 0 and 1")
    courses = sorted(set(lecture_courses.values()))
    if len(courses) >= MIN_COURSES_FOR_COURSE_SPLIT:
        test_courses = _held_out(courses, test_fraction, seed, "course")
        test = sorted(lec for lec, c in lecture_courses.items() if c in test_courses)
        unit = "course"
    else:
        test = sorted(_held_out(list(lecture_courses), test_fraction, seed, "lecture"))
        unit = "lecture"
    test_set = set(test)
    train = sorted(lec for lec in lecture_courses if lec not in test_set)
    split = {
        "version": SPLIT_VERSION,
        "unit": unit,
        "seed": seed,
        "test_fraction": test_fraction,
        "train": train,
        "test": test,
        "courses": {lec: lecture_courses[lec] for lec in sorted(lecture_courses)},
    }
    validate_split(split)
    return split


def validate_split(split: dict, lecture_courses: Mapping[str, str] | None = None) -> None:
    """Raise ``LeakageError``/``SplitError`` unless the split is disjoint, complete and grouped."""
    train, test = set(split["train"]), set(split["test"])
    if not train or not test:
        raise SplitError("train and test must both be non-empty")
    if train & test:
        raise LeakageError(f"lectures in both train and test: {sorted(train & test)}")
    courses = split["courses"]
    if set(courses) != train | test:
        raise SplitError("split courses do not cover exactly the train and test lectures")
    if lecture_courses is not None and dict(lecture_courses) != courses:
        raise SplitError("the split does not match the given lectures and courses")
    if split["unit"] == "course":
        shared = {courses[lec] for lec in train} & {courses[lec] for lec in test}
        if shared:
            raise LeakageError(f"courses in both train and test: {sorted(shared)}")


def write_split_once(path: Path, split: dict) -> None:
    """Write ``split`` as JSON; refuses to overwrite an existing file (the test set is sealed)."""
    validate_split(split)
    try:
        with open(path, "x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(split, sort_keys=True, indent=2) + "\n")
    except FileExistsError as exc:
        raise SplitError(f"{path} already exists: the sealed split is never overwritten") from exc


def read_split(path: Path) -> dict:
    """Read and validate a sealed split."""
    try:
        split = json.loads(Path(path).read_text(encoding="utf-8"))
        version = split["version"]
        validate_split(split)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SplitError(f"cannot use split file {path}: {exc}") from exc
    if version != SPLIT_VERSION:
        raise SplitError(f"unsupported split version {version!r}")
    return split


def group_kfold(lectures: Iterable[str], k: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """``k`` (train, validation) lecture lists; every lecture is validated exactly once."""
    ids = sorted(set(lectures))
    if k < 2 or k > len(ids):
        raise SplitError(f"k must be between 2 and the number of lectures ({len(ids)}), got {k}")
    random.Random(seed).shuffle(ids)
    folds = [sorted(ids[i::k]) for i in range(k)]
    out = []
    for i, valid in enumerate(folds):
        train = sorted(lec for j, fold in enumerate(folds) if j != i for lec in fold)
        check_disjoint(train, valid)
        out.append((train, valid))
    return out


def check_disjoint(train: Iterable[str], held_out: Iterable[str]) -> None:
    """Raise ``LeakageError`` when a lecture is on both sides."""
    both = set(train) & set(held_out)
    if both:
        raise LeakageError(f"lectures on both sides: {sorted(both)}")
