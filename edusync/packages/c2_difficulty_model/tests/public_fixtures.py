"""Tiny in-memory workbooks and OneStop archives for the public-data tests (no real downloads)."""

import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"


def col_letters(index: int) -> str:
    out = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def make_xlsx(
    path: Path,
    rows: list[list],
    *,
    sheet_path: str = "xl/worksheets/sheet1.xml",
    rels_target: str | None = "worksheets/sheet1.xml",
    shared: bool = True,
    rel_id: str = "rId1",
) -> Path:
    """Write ``rows`` (str, int/float or None) as the first worksheet of a minimal workbook."""
    strings: list[str] = []
    cells_xml = []
    for r, row in enumerate(rows, start=1):
        cells = []
        for c, value in enumerate(row):
            ref = f"{col_letters(c)}{r}"
            if value is None:
                continue
            if isinstance(value, str) and shared:
                strings.append(value)
                cells.append(f'<c r="{ref}" t="s"><v>{len(strings) - 1}</v></c>')
            elif isinstance(value, str):
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
        cells_xml.append(f'<row r="{r}">{"".join(cells)}</row>')
    sheet = f'<worksheet xmlns="{MAIN}"><sheetData>{"".join(cells_xml)}</sheetData></worksheet>'
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(sheet_path, sheet)
        if rels_target is not None:
            archive.writestr(
                "xl/workbook.xml",
                f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
                f'<sheet name="S" sheetId="1" r:id="rId1"/></sheets></workbook>',
            )
            archive.writestr(
                "xl/_rels/workbook.xml.rels",
                f'<Relationships xmlns="{PKG}"><Relationship Id="{rel_id}" '
                f'Target="{rels_target}"/></Relationships>',
            )
        if shared:
            items = []
            for i, text in enumerate(strings):
                body = escape(text)
                if i % 2:  # exercise rich-text runs
                    half = len(body) // 2
                    items.append(f"<si><r><t>{body[:half]}</t></r><r><t>{body[half:]}</t></r></si>")
                else:
                    items.append(f"<si><t>{body}</t></si>")
            archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{MAIN}">{"".join(items)}</sst>')
    return path


def clear_rows(n: int = 30) -> list[list]:
    """Header + ``n`` passages; ease rises with the id so difficulty falls."""
    rows: list[list] = [["ID", "Author", "Excerpt", "BT_easiness", "s.e."]]
    for i in range(n):
        text = "The cat sat on the mat. " * (1 + i % 3) + f"Passage number {i}."
        rows.append([str(i), "A. Writer", text, (i - n / 2) / 10, 0.4])
    return rows


def make_onestop_zip(path: Path, n_articles: int = 12) -> Path:
    """Three levels per article, plus the junk the real archive contains."""
    base = "OneStopEnglishCorpus-master/Texts-SeparatedByReadingLevel"
    words = {"Ele": "Dogs run.", "Int": "Dogs run quickly home.", "Adv": "Canines sprint rapidly."}
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n_articles):
            for level, text in words.items():
                body = ("﻿" if i == 0 else "") + f"{text} Story {i}. " * 3
                archive.writestr(f"{base}/{level}-Txt/Article {i} -{level.lower()}.txt", body)
        archive.writestr(f"{base}/Int-Txt/Int-Txt/Article 0 -int.txt", "nested duplicate")
        archive.writestr(f"{base}/Ele-Txt/.DS_Store", "junk")
        archive.writestr(f"{base}/Ele-Txt/.hidden.txt", "junk")
        archive.writestr(f"{base}/Ele-Txt/notes.md", "junk")
        archive.writestr("OneStopEnglishCorpus-master/README.md", "readme")
        archive.writestr("short.txt", "x")
    return path
