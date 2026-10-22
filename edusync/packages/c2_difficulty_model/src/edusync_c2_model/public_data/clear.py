"""Loader for the CLEAR corpus workbook (CC BY-NC-SA 4.0: research use only, never redistributed).

Columns used (exact header names in ``CLEAR_corpus_final.xlsx``): ``ID`` (group id, one passage
per id), ``Excerpt`` (text) and ``BT_easiness`` (Bradley-Terry ease, higher = EASIER). The
precomputed readability columns in the file are deliberately ignored.
"""

import math
from pathlib import Path

from edusync_c2_model.public_data.schema import SOURCE_CLEAR, TextExample
from edusync_c2_model.public_data.xlsx import read_table

COL_ID = "ID"
COL_TEXT = "Excerpt"
COL_EASE = "BT_easiness"


class ClearError(ValueError):
    """The workbook does not look like the CLEAR corpus."""


def parse_rows(table: list[dict[str, str]]) -> list[TextExample]:
    """Examples of every row with a non-empty excerpt; target is the raw ease (higher = easier)."""
    if table:
        missing = {COL_ID, COL_TEXT, COL_EASE} - set(table[0])
        if missing:
            raise ClearError(f"missing columns: {sorted(missing)}")
    examples = []
    seen: set[str] = set()
    for row in table:
        text = row[COL_TEXT].strip()
        if not text:
            continue
        group_id = row[COL_ID].strip()
        if group_id in seen:
            raise ClearError(f"duplicate passage id {group_id!r}")
        seen.add(group_id)
        try:
            ease = float(row[COL_EASE])
        except ValueError as exc:
            raise ClearError(f"passage {group_id!r} has no numeric {COL_EASE}") from exc
        if not math.isfinite(ease):
            raise ClearError(f"passage {group_id!r} has a non-finite {COL_EASE}")
        examples.append(
            TextExample(source=SOURCE_CLEAR, text=text, target=ease, level=None, group_id=group_id)
        )
    return examples


def load_clear(path: Path) -> list[TextExample]:
    """Read the first worksheet of ``path`` and return its examples, in file order."""
    return parse_rows(read_table(Path(path)))
