"""The lecture manifest: which files make up a dataset and which course each lecture belongs to.

CSV columns: ``lecture_id, course_id, file`` and optionally ``graph`` (a C1 graph JSON). Relative
paths are resolved against the manifest's folder. Ids are pseudonyms: letters, digits, ``-`` and
``_``, at most 32 characters.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from edusync_c2_model.ids import is_valid_id

REQUIRED_COLUMNS = ("lecture_id", "course_id", "file")


class ManifestError(ValueError):
    """The manifest is malformed or points at something that does not exist."""


@dataclass(frozen=True)
class ManifestEntry:
    lecture_id: str
    course_id: str
    file: Path
    graph: Path | None = None


def _resolve(base: Path, text: str, line: int, what: str) -> Path:
    path = Path(text)
    if not path.is_absolute():
        path = base / path
    if not path.is_file():
        raise ManifestError(f"line {line}: {what} file does not exist: {text}")
    return path


def read_manifest(path: Path) -> list[ManifestEntry]:
    """Validated entries in file order."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = [h.strip() for h in (reader.fieldnames or [])]
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                raise ManifestError(f"manifest is missing column(s): {', '.join(missing)}")
            raw_rows = [
                (reader.line_num, {(k or "").strip(): (v or "").strip() for k, v in raw.items()})
                for raw in reader
            ]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc

    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for line, row in raw_rows:
        for column in ("lecture_id", "course_id"):
            if not is_valid_id(row[column]):
                raise ManifestError(
                    f"line {line}: {column} {row[column]!r} is not a valid id "
                    "(1-32 letters, digits, '-' or '_')"
                )
        lecture = row["lecture_id"]
        if lecture in seen:
            raise ManifestError(f"line {line}: duplicate lecture_id {lecture!r}")
        seen.add(lecture)
        graph_text = row.get("graph", "")
        entries.append(
            ManifestEntry(
                lecture_id=lecture,
                course_id=row["course_id"],
                file=_resolve(path.parent, row["file"], line, "lecture"),
                graph=_resolve(path.parent, graph_text, line, "graph") if graph_text else None,
            )
        )
    if not entries:
        raise ManifestError("manifest has no lectures")
    return entries


def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    """Write ``entries`` with paths relative to the manifest's folder (posix separators)."""
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("lecture_id", "course_id", "file", "graph"))
        for e in entries:
            graph = e.graph.relative_to(path.parent).as_posix() if e.graph else ""
            writer.writerow(
                (e.lecture_id, e.course_id, e.file.relative_to(path.parent).as_posix(), graph)
            )
