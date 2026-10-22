"""Minimal stdlib reader for the first worksheet of an ``.xlsx`` file (no openpyxl).

Only what the public corpora need: shared strings, inline strings, numbers and the cell
references that place a value in its column. Every cell is returned as ``str`` (numbers keep the
text stored in the file) so callers decide how to parse them.
"""

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
FALLBACK_SHEET = "xl/worksheets/sheet1.xml"
_CELL_REF = re.compile(r"([A-Z]+)")


class XlsxError(ValueError):
    """The file is not a readable workbook."""


def _text(element: ElementTree.Element) -> str:
    """Concatenated ``<t>`` text of a shared or inline string (rich-text runs included)."""
    return "".join(t.text or "" for t in element.iter(f"{NS}t"))


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [_text(si) for si in root.iter(f"{NS}si")]


def _first_sheet_path(archive: zipfile.ZipFile) -> str:
    """Path of the first worksheet in workbook order; ``sheet1.xml`` when it cannot be resolved."""
    try:
        workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        rels = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        sheet = next(workbook.iter(f"{NS}sheet"))
        rel_id = sheet.get(f"{REL_NS}id")
        for rel in rels.iter(f"{PKG_REL_NS}Relationship"):
            if rel.get("Id") == rel_id:
                target = rel.get("Target", "").lstrip("/")
                return target if target.startswith("xl/") else f"xl/{target}"
    except (KeyError, StopIteration, ElementTree.ParseError):
        pass
    return FALLBACK_SHEET


def _column_index(ref: str) -> int:
    letters = _CELL_REF.match(ref)
    if letters is None:
        raise XlsxError(f"bad cell reference {ref!r}")
    index = 0
    for char in letters.group(1):
        index = index * 26 + (ord(char) - 64)
    return index - 1


def _cell_value(cell: ElementTree.Element, strings: list[str]) -> str:
    kind = cell.get("t")
    if kind == "inlineStr":
        return _text(cell)
    value = cell.find(f"{NS}v")
    if value is None or value.text is None:
        return ""
    if kind == "s":
        return strings[int(value.text)]
    return value.text


def read_rows(path: Path) -> list[list[str]]:
    """All rows of the first worksheet as lists of strings (empty cells are ``""``)."""
    try:
        with zipfile.ZipFile(path) as archive:
            strings = _shared_strings(archive)
            sheet = ElementTree.fromstring(archive.read(_first_sheet_path(archive)))
    except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise XlsxError(f"cannot read workbook {path}: {type(exc).__name__}") from exc
    rows: list[list[str]] = []
    for row in sheet.iter(f"{NS}row"):
        cells: dict[int, str] = {}
        for cell in row.iter(f"{NS}c"):
            cells[_column_index(cell.get("r", ""))] = _cell_value(cell, strings)
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])
    return rows


def read_table(path: Path) -> list[dict[str, str]]:
    """First row is the header; each later row becomes a dict (short rows are padded with "")."""
    rows = read_rows(path)
    if not rows:
        raise XlsxError(f"{path} has no rows")
    header = rows[0]
    return [
        {name: (row[i] if i < len(row) else "") for i, name in enumerate(header)}
        for row in rows[1:]
    ]
