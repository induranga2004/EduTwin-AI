"""Seeded 80/10/10 train/val/test split of public text examples, grouped by ``group_id``.

A CLEAR passage is its own group; the three levels of a OneStop article share one group, so they
never straddle splits. Group ids are de-duplicated and sorted before the seeded shuffle, so the
result depends only on the ids and the seed. The sealed split is written once and only read after.
"""

import json
import random
from collections.abc import Iterable
from pathlib import Path

from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.split import LeakageError, SplitError

PUBLIC_SPLIT_VERSION = 1
PARTS = ("train", "val", "test")
MIN_GROUPS = 3


def make_group_split(
    group_ids: Iterable[str],
    *,
    seed: int,
    source: str,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict:
    """``{"train", "val", "test"}`` lists of group ids plus the settings that produced them."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise SplitError("val and test fractions must be in (0, 1) and sum to less than 1")
    ids = sorted(set(group_ids))
    if len(ids) < MIN_GROUPS:
        raise SplitError(f"need at least {MIN_GROUPS} groups to split, got {len(ids)}")
    random.Random(seed).shuffle(ids)
    n_test = max(1, round(test_fraction * len(ids)))
    n_val = max(1, round(val_fraction * len(ids)))
    split = {
        "version": PUBLIC_SPLIT_VERSION,
        "source": source,
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "test": sorted(ids[:n_test]),
        "val": sorted(ids[n_test : n_test + n_val]),
        "train": sorted(ids[n_test + n_val :]),
    }
    validate_group_split(split)
    return split


def validate_group_split(split: dict) -> None:
    """Raise ``SplitError``/``LeakageError`` unless the three parts are non-empty and disjoint."""
    sets = {name: set(split[name]) for name in PARTS}
    if not all(sets.values()):
        raise SplitError("train, val and test must all be non-empty")
    for i, first in enumerate(PARTS):
        for second in PARTS[i + 1 :]:
            both = sets[first] & sets[second]
            if both:
                raise LeakageError(f"groups in both {first} and {second}: {sorted(both)[:5]}")


def write_split_once(path: Path, split: dict) -> None:
    """Write the split as JSON; refuses to overwrite (the test split is sealed)."""
    validate_group_split(split)
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
        validate_group_split(split)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SplitError(f"cannot use split file {path}: {exc}") from exc
    if version != PUBLIC_SPLIT_VERSION:
        raise SplitError(f"unsupported split version {version!r}")
    return split


def partition(examples: Iterable[TextExample], split: dict) -> dict[str, list[TextExample]]:
    """Examples per part, in input order; a group the split does not know is an error."""
    owner = {gid: name for name in PARTS for gid in split[name]}
    out: dict[str, list[TextExample]] = {name: [] for name in PARTS}
    for example in examples:
        name = owner.get(example.group_id)
        if name is None:
            raise SplitError(f"group {example.group_id!r} is not in the split")
        out[name].append(example)
    return out
