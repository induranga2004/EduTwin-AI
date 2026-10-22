"""Lecture materials parser for the difficulty pipeline (PPTX and PDF).

C2 owns its parser because C1's extractor flattens text and loses titles, speaker notes, bullet
levels, pictures, charts and tables. A PPTX keeps all of that; a PDF only yields text per page,
so its structure fields stay ``None`` and ``structure_available`` is ``False``.
"""

import re
from io import BytesIO

from pptx import Presentation
from pptx.shapes.group import GroupShape
from pptx.shapes.picture import Picture
from pptx.shapes.placeholder import PlaceholderPicture
from pptx.slide import Slide
from pptx.text.text import TextFrame
from pypdf import PdfReader

from edusync_c2.difficulty.schemas import MaterialsDoc, ParagraphRecord, SlideRecord

ALLOWED_EXTENSIONS = (".pptx", ".pdf")
PDF_STRUCTURE_WARNING = "PDF: structure fields (pictures, charts, bullet levels) are unavailable"


class MaterialsError(Exception):
    """A rejected upload, carrying the HTTP status the API should answer with."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


# ------------------------------------------------------------------------------------ detectors

_LATEX = re.compile(r"\\(?:frac|sum|int|sqrt)(?![A-Za-z])|\$\$")
_MATH_SYMBOLS = ("\u2211", "\u222b", "\u221a", "\u2264", "\u2265", "\u2202", "\u221e", "\u00b1")
_COMPARISONS = re.compile(r"==|=>|<=|>=|!=")
_EQUATION = re.compile(r"[\w)\]]\s*=\s*[\w(\-]")
_LONG_WORD = re.compile(r"[A-Za-z]{4,}")
_KEYWORDS = re.compile(r"\b(?:def|class|import|function)\s+\w|\breturn[ ;]")
_SQL = re.compile(r"\bSELECT\b.*\bFROM\b", re.IGNORECASE)
_CODE_OPERATORS = ("==", "=>", "->")
_INDENTED = re.compile(r"^(?: {4}|\t)\S", re.MULTILINE)


def detect_formula(text: str) -> bool:
    """Heuristic: does the text contain a formula or equation?

    True when any of these holds:
    - a LaTeX marker (``\\frac``, ``\\sum``, ``\\int``, ``\\sqrt``, ``$$``);
    - a math symbol (Unicode sum, integral, root, less/greater-or-equal, partial, infinity,
      plus-minus) or the ASCII ``<=`` / ``>=``;
    - an equation-like line: a single ``=`` (not part of ``==``, ``=>``, ``<=``, ``>=``, ``!=``)
      between operands, with an operator, parenthesis or exponent on the line, at most one word
      of four or more letters (so prose sentences do not count), and not ending in ``;`` (code).
    """
    if _LATEX.search(text) or any(s in text for s in (*_MATH_SYMBOLS, "<=", ">=")):
        return True
    for line in text.splitlines():
        line = line.strip()
        if not line or line.endswith(";") or len(line) > 80:
            continue
        bare = _COMPARISONS.sub(" ", line)
        has_operator = any(c in bare for c in "+*/^()") or " - " in bare
        if _EQUATION.search(bare) and has_operator and len(_LONG_WORD.findall(bare)) <= 1:
            return True
    return False


def detect_code(text: str) -> bool:
    """Heuristic: does the text look like source code?

    True when at least two of these signals are present: a line ending in ``;``; a keyword
    (``def x``, ``class x``, ``import x``, ``function x``, ``return``, ``SELECT ... FROM``);
    braces; one of ``==``, ``=>``, ``->``; a line indented by four spaces or a tab.
    """
    signals = (
        any(line.rstrip().endswith(";") for line in text.splitlines()),
        bool(_KEYWORDS.search(text) or _SQL.search(text)),
        "{" in text or "}" in text,
        any(op in text for op in _CODE_OPERATORS),
        bool(_INDENTED.search(text)),
    )
    return sum(signals) >= 2


# ------------------------------------------------------------------------------------- parsing


def parse_materials(data: bytes, filename: str, *, max_mb: int = 100) -> MaterialsDoc:
    """Validate and parse an uploaded PPTX or PDF into a ``MaterialsDoc``.

    Raises ``MaterialsError``: 400 (unsupported type, empty, wrong signature, unreadable,
    encrypted PDF), 413 (larger than ``max_mb`` MB), 422 (no text on any slide or page).
    """
    name = filename.lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise MaterialsError(400, "Unsupported file type; upload a .pptx or .pdf file")
    if not data:
        raise MaterialsError(400, "Uploaded file is empty")
    if len(data) > max_mb * 1024 * 1024:
        raise MaterialsError(413, f"File exceeds the {max_mb} MB limit")
    if name.endswith(".pdf"):
        if not data.startswith(b"%PDF-"):
            raise MaterialsError(400, "File content is not a PDF")
        doc = _parse_pdf(data, filename)
    else:
        if not data.startswith(b"PK"):
            raise MaterialsError(400, "File content is not a PPTX")
        doc = _parse_pptx(data, filename)
    if not any(slide.text.strip() for slide in doc.slides):
        raise MaterialsError(422, "No extractable text found in the file")
    return doc


def _index_list(indexes: list[int]) -> str:
    return ", ".join(str(i) for i in indexes)


def _parse_pdf(data: bytes, filename: str) -> MaterialsDoc:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise MaterialsError(400, "Encrypted PDFs are not supported")
        texts = [(page.extract_text() or "").strip() for page in reader.pages]
    except MaterialsError:
        raise
    except Exception as exc:
        raise MaterialsError(400, "Could not read the PDF") from exc
    slides = [
        SlideRecord(
            index=i,
            text=text,
            paragraphs=[
                ParagraphRecord(text=line.strip()) for line in text.splitlines() if line.strip()
            ],
            formula_hint=detect_formula(text),
            code_hint=detect_code(text),
        )
        for i, text in enumerate(texts, start=1)
    ]
    warnings = [PDF_STRUCTURE_WARNING]
    blank = [s.index for s in slides if not s.text]
    if blank:
        warnings.append(f"PDF pages without extractable text: {_index_list(blank)}")
    return MaterialsDoc(
        filename=filename,
        file_type="pdf",
        slide_count=len(slides),
        structure_available=False,
        slides=slides,
        warnings=warnings,
    )


class _SlideCollector:
    """Accumulates paragraphs and shape counts while walking a slide's shape tree."""

    def __init__(self) -> None:
        self.paragraphs: list[ParagraphRecord] = []
        self.pictures = self.charts = self.tables = self.shapes = 0

    def add_frame(self, frame: TextFrame) -> None:
        for paragraph in frame.paragraphs:
            text = paragraph.text.replace("\v", " ").rstrip()
            if text.strip():
                self.paragraphs.append(ParagraphRecord(text=text, level=paragraph.level))

    def walk(self, shapes, title_id: int | None) -> None:
        for shape in shapes:
            self.shapes += 1
            if shape.shape_id == title_id:
                continue
            # isinstance, not shape.shape_type: that raises NotImplementedError for an autoshape
            # without recognisable geometry, which would turn a valid deck into an HTTP 400.
            if isinstance(shape, GroupShape):
                self.walk(shape.shapes, title_id)
                continue
            # A filled picture placeholder is a PlaceholderPicture (shape_type PLACEHOLDER).
            if isinstance(shape, Picture | PlaceholderPicture):
                self.pictures += 1
            if shape.has_chart:
                self.charts += 1
            if shape.has_table:
                self.tables += 1
                for row in shape.table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    if any(cells):
                        self.paragraphs.append(ParagraphRecord(text=" | ".join(cells), level=0))
            elif shape.has_text_frame:
                self.add_frame(shape.text_frame)


def _parse_slide(slide: Slide, index: int) -> SlideRecord:
    collector = _SlideCollector()
    title_shape = slide.shapes.title
    title = None
    if title_shape is not None:
        collector.add_frame(title_shape.text_frame)
        title = title_shape.text_frame.text.strip() or None
    collector.walk(slide.shapes, title_shape.shape_id if title_shape is not None else None)
    text = "\n".join(p.text for p in collector.paragraphs)
    notes = ""
    if slide.has_notes_slide:
        notes = (slide.notes_slide.notes_text_frame.text or "").strip()
    return SlideRecord(
        index=index,
        title=title,
        text=text,
        notes=notes,
        paragraphs=collector.paragraphs,
        picture_count=collector.pictures,
        chart_count=collector.charts,
        table_count=collector.tables,
        shape_count=collector.shapes,
        formula_hint=detect_formula(text),
        code_hint=detect_code(text),
    )


def _parse_pptx(data: bytes, filename: str) -> MaterialsDoc:
    try:
        presentation = Presentation(BytesIO(data))
        slides = [_parse_slide(s, i) for i, s in enumerate(presentation.slides, start=1)]
    except Exception as exc:
        raise MaterialsError(400, "Could not read the PPTX") from exc
    warnings = []
    empty = [s.index for s in slides if not s.text.strip()]
    if empty:
        warnings.append(f"Slides without text: {_index_list(empty)}")
    return MaterialsDoc(
        filename=filename,
        file_type="pptx",
        slide_count=len(slides),
        structure_available=True,
        slides=slides,
        warnings=warnings,
    )
