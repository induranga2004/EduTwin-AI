"""Loader for OneStopEnglish (CC BY-SA 4.0), read straight from the downloaded zip.

Layout: ``<root>/Texts-SeparatedByReadingLevel/{Ele,Int,Adv}-Txt/<article>-<ele|int|adv>.txt``.
The archive also holds a nested duplicate ``Int-Txt/Int-Txt`` folder and macOS junk; only files
sitting directly in a level folder are read. Level 0/1/2 = Ele/Int/Adv (higher = harder); the
group id is the article name, so the three versions of one article stay together in a split.
"""

import re
import zipfile
from pathlib import Path

from edusync_c2_model.public_data.schema import SOURCE_ONESTOP, TextExample

LEVEL_DIRS = {"Ele-Txt": 0, "Int-Txt": 1, "Adv-Txt": 2}
TEXTS_DIR = "Texts-SeparatedByReadingLevel"
_SUFFIX = re.compile(r"\s*-(?:ele|int|adv)$", re.IGNORECASE)


class OneStopError(ValueError):
    """The archive is unreadable or does not have the expected layout."""


def _classify(name: str) -> tuple[int, str] | None:
    parts = name.split("/")
    if len(parts) < 3 or parts[-3] != TEXTS_DIR or parts[-2] not in LEVEL_DIRS:
        return None
    filename = parts[-1]
    if not filename.endswith(".txt") or filename.startswith("."):
        return None
    article = _SUFFIX.sub("", filename[: -len(".txt")]).strip()
    return LEVEL_DIRS[parts[-2]], article


def load_onestop(path: Path) -> list[TextExample]:
    """All texts, sorted by (group_id, level) so the order never depends on the zip."""
    try:
        with zipfile.ZipFile(path) as archive:
            found = [(n, _classify(n)) for n in archive.namelist()]
            examples = []
            for name, info in found:
                if info is None:
                    continue
                level, article = info
                text = archive.read(name).decode("utf-8-sig").strip()
                examples.append(
                    TextExample(
                        source=SOURCE_ONESTOP,
                        text=text,
                        target=float(level),
                        level=level,
                        group_id=article,
                    )
                )
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
        raise OneStopError(f"cannot read {path}: {type(exc).__name__}") from exc
    keys = [(e.group_id, e.level) for e in examples]
    if not examples or len(set(keys)) != len(keys):
        raise OneStopError("no texts found, or an article/level pair occurs twice")
    return sorted(examples, key=lambda e: (e.group_id, e.level))
