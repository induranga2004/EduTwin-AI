"""Tests for the PPTX/PDF materials parser and the formula/code detectors."""

from io import BytesIO

import pytest
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from pptx.util import Inches
from pypdf import PdfWriter

from edusync_c2.difficulty.materials import (
    MaterialsError,
    detect_code,
    detect_formula,
    parse_materials,
)
from edusync_c2.difficulty.synthetic import _png, build_text_pdf


def _save(prs: Presentation) -> bytes:
    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()


@pytest.fixture
def rich_pptx() -> bytes:
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[5])
    s1.shapes.title.text = "Intro"
    box = s1.shapes.add_textbox(Inches(1), Inches(2), Inches(5), Inches(2))
    box.text_frame.text = "Top line"
    nested = box.text_frame.add_paragraph()
    nested.text = "Nested line"
    nested.level = 2
    box.text_frame.add_paragraph()  # empty paragraph is skipped
    brk = box.text_frame.add_paragraph()
    brk.text = "first"
    brk.add_line_break()
    brk.add_run().text = "second"
    s1.shapes.add_picture(BytesIO(_png()), Inches(6), Inches(2), Inches(1))
    table = s1.shapes.add_table(3, 2, Inches(1), Inches(4), Inches(4), Inches(1)).table
    table.cell(0, 0).text = "a"
    table.cell(0, 1).text = "b"
    table.cell(2, 0).text = "c"  # row 1 stays blank and is skipped
    group = s1.shapes.add_group_shape()
    group.shapes.add_textbox(Inches(1), Inches(6), Inches(2), Inches(1)).text_frame.text = "grouped"
    group.shapes.add_picture(BytesIO(_png()), Inches(4), Inches(6), Inches(1))
    s1.notes_slide.notes_text_frame.text = "speaker notes"

    s2 = prs.slides.add_slide(prs.slide_layouts[6])  # no title placeholder
    data = CategoryChartData()
    data.categories = ["x", "y"]
    data.add_series("s", (1, 2))
    s2.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(1), Inches(1), Inches(4), Inches(3), data
    )
    s2.shapes.add_textbox(
        Inches(1), Inches(5), Inches(4), Inches(1)
    ).text_frame.text = "E = m * (c^2)"

    prs.slides.add_slide(prs.slide_layouts[6])  # empty slide
    return _save(prs)


def test_pptx_structure(rich_pptx: bytes) -> None:
    doc = parse_materials(rich_pptx, "Deck.PPTX")
    assert doc.file_type == "pptx"
    assert doc.structure_available
    assert doc.slide_count == len(doc.slides) == 3
    first, second, third = doc.slides
    assert first.index == 1
    assert first.title == "Intro"
    assert first.text.splitlines()[0] == "Intro"
    assert "a | b" in first.text
    assert "c |" in first.text
    assert "grouped" in first.text
    assert "first second" in first.text
    assert first.notes == "speaker notes"
    assert (first.picture_count, first.chart_count, first.table_count) == (2, 0, 1)
    # title, textbox, picture, table, group + its 2 children
    assert first.shape_count == 7
    levels = {p.text: p.level for p in first.paragraphs}
    assert levels["Top line"] == 0
    assert levels["Nested line"] == 2
    assert all(p.text.strip() for p in first.paragraphs)
    assert second.title is None
    assert second.chart_count == 1
    assert second.formula_hint
    assert not second.code_hint
    assert third.text == ""
    assert third.shape_count == 0
    assert doc.warnings == ["Slides without text: 3"]


def test_pdf_pages_and_blank_page_warning() -> None:
    data = build_text_pdf([["Title one", "E = m * (c^2)"], [], ["def f(x):", "    return x;"]])
    doc = parse_materials(data, "notes.pdf")
    assert doc.file_type == "pdf"
    assert not doc.structure_available
    assert doc.slide_count == len(doc.slides) == 3
    first, blank, third = doc.slides
    assert first.title is None
    assert first.picture_count is None
    assert first.shape_count is None
    assert [p.text for p in first.paragraphs] == ["Title one", "E = m * (c^2)"]
    assert all(p.level is None for p in first.paragraphs)
    assert first.formula_hint
    assert third.code_hint
    assert blank.text == ""
    assert blank.paragraphs == []
    assert doc.warnings[0].startswith("PDF: structure fields")
    assert doc.warnings[1] == "PDF pages without extractable text: 2"


def _blank_pdf(encrypt: bool = False) -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    if encrypt:
        writer.encrypt("secret")
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_pdf_without_text_is_422() -> None:
    with pytest.raises(MaterialsError) as err:
        parse_materials(_blank_pdf(), "x.pdf")
    assert err.value.status_code == 422


def test_pptx_without_text_is_422() -> None:
    prs = Presentation()
    prs.slides.add_slide(prs.slide_layouts[6])
    with pytest.raises(MaterialsError) as err:
        parse_materials(_save(prs), "x.pptx")
    assert err.value.status_code == 422


@pytest.mark.parametrize(
    ("data", "name", "fragment"),
    [
        (b"hello", "x.txt", "Unsupported"),
        (b"", "x.pdf", "empty"),
        (b"PK\x03\x04", "x.pdf", "not a PDF"),
        (b"%PDF-1.4", "x.pptx", "not a PPTX"),
        (b"%PDF-1.4 garbage", "x.pdf", "Could not read the PDF"),
        (b"PK\x03\x04 garbage", "x.pptx", "Could not read the PPTX"),
    ],
)
def test_validation_errors(data: bytes, name: str, fragment: str) -> None:
    with pytest.raises(MaterialsError) as err:
        parse_materials(data, name)
    assert err.value.status_code == 400
    assert fragment in err.value.message
    assert str(err.value) == err.value.message


def test_encrypted_pdf_rejected() -> None:
    with pytest.raises(MaterialsError) as err:
        parse_materials(_blank_pdf(encrypt=True), "x.pdf")
    assert err.value.status_code == 400
    assert "Encrypted" in err.value.message


def test_too_large_is_413() -> None:
    with pytest.raises(MaterialsError) as err:
        parse_materials(b"%PDF-" + b"0" * (2 * 1024 * 1024), "x.pdf", max_mb=1)
    assert err.value.status_code == 413


@pytest.mark.parametrize(
    "text",
    [
        "x = \\frac{a}{b}",
        "$$ a + b $$",
        "\\sum_i x_i",
        "total ∑ values",
        "a ≤ b",
        "value >= 3",
        "L(w) = sum((y - w*x)^2) / n",
        "y = a*x^2 + b*x + c",
        "z = (x - mu) / sigma",
        "area = 3 - 1",
    ],
)
def test_detect_formula_true(text: str) -> None:
    assert detect_formula(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Gradient descent minimises the loss function.",
        "x = 5",
        "The result = good (usually) when the data is clean",
        "total = a + b;",
        "if a == b then stop",
        "x = " + "1 + " * 30 + "1",
        "-----",
    ],
)
def test_detect_formula_false(text: str) -> None:
    assert not detect_formula(text)


@pytest.mark.parametrize(
    "text",
    [
        "def run(x):\n    return x;",
        "import os\nvalue = {1: 2}",
        "function f() {\n  return 1;\n}",
        "SELECT name FROM users\nwhere a == b",
        "x -> y;\n\tprint(x)",
    ],
)
def test_detect_code_true(text: str) -> None:
    assert detect_code(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Plain prose about classes and functions.",
        "Please return the form",
        "SELECT only one signal here",
        "y = a*x^2 + b*x + c",
    ],
)
def test_detect_code_false(text: str) -> None:
    assert not detect_code(text)
