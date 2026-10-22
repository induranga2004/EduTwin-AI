"""Lecture report: slides with the lecturer's drawings on top, times, and what was written.

Pure functions that turn stored C4 data into one self-contained HTML page (inline SVG, inline
CSS, no scripts, no external requests). It prints cleanly to PDF from any browser.

Coordinates: BBB places the slide image on the whiteboard at (0, 0) with size
`scaledWidth x scaledHeight` and annotations use that same space (checked in
bigbluebutton-html5 `whiteboard/container.jsx`). The slide SVG is therefore stretched to that
size and shapes are drawn at their own coordinates.

Everything that comes from the lecture (text, ids) is escaped. The slide image is embedded as a
base64 data URI inside an SVG <image>, which browsers never execute scripts from.
"""

import base64
import math
import textwrap
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from typing import Any

from edusync_c4.diagram import build_diagrams
from edusync_c4.schemas import (
    AnnotationGroup,
    AnnotationTimeline,
    Diagram,
    NormalizedAnnotation,
    ShapeType,
    SlideInfo,
)
from edusync_c4.timeline import reconstruct_snapshot

DEFAULT_SIZE = (1440.0, 810.0)

COLORS = {
    "black": "#1d1d1d",
    "grey": "#8e9ba8",
    "light-violet": "#d27ceb",
    "violet": "#ae3ec9",
    "blue": "#4465e9",
    "light-blue": "#4ba1f1",
    "yellow": "#f1ac4b",
    "orange": "#e16919",
    "green": "#099268",
    "light-green": "#4cb05e",
    "light-red": "#f87777",
    "red": "#e03131",
    "white": "#ffffff",
}
NOTE_FILL = {
    "black": "#fce19c",
    "grey": "#d9dde2",
    "light-violet": "#ebc8f7",
    "violet": "#dbb8f0",
    "blue": "#aeccfa",
    "light-blue": "#bddcfa",
    "yellow": "#ffe27a",
    "orange": "#ffc08a",
    "green": "#a6dcb2",
    "light-green": "#c2e8c8",
    "light-red": "#fbc4c4",
    "red": "#f7a5a5",
    "white": "#f7f7f7",
}
STROKE = {"s": 2.0, "m": 3.5, "l": 5.0, "xl": 10.0}
TEXT_SIZE = {"s": 18.0, "m": 24.0, "l": 36.0, "xl": 44.0}
LABEL_SIZE = {"s": 18.0, "m": 22.0, "l": 26.0, "xl": 32.0}
TYPE_LABEL = {
    ShapeType.TEXT: "Text box",
    ShapeType.NOTE: "Sticky note",
    ShapeType.DRAW: "Freehand drawing",
    ShapeType.HIGHLIGHT: "Highlight",
    ShapeType.ARROW: "Arrow",
    ShapeType.LINE: "Line",
    ShapeType.GEO: "Shape",
    ShapeType.FRAME: "Frame",
    ShapeType.POLL: "Poll",
    ShapeType.UNKNOWN: "Item",
}


@dataclass
class SlideView:
    slide_id: str
    info: SlideInfo | None
    svg: str | None
    timeline: AnnotationTimeline
    groups: list[AnnotationGroup]


# ---------------------------------------------------------------------------- helpers


def fmt(n: float) -> str:
    return f"{n:.1f}".rstrip("0").rstrip(".") if n != int(n) else str(int(n))


def clock(seconds: float) -> str:
    s = max(0, round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def wrap_text(text: str, width_px: float, font_px: float) -> list[str]:
    """Greedy word wrap by estimated character width (no font metrics available)."""
    per_line = max(4, int(width_px / (font_px * 0.55)))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        lines.extend(
            textwrap.wrap(
                paragraph, width=per_line, break_long_words=True, replace_whitespace=False
            )
            or [""]
        )
    return lines


def _color(props: dict[str, Any], key: str = "color") -> str:
    return COLORS.get(str(props.get(key, "black")), COLORS["black"])


def _point(p: Any) -> tuple[float, float]:
    if isinstance(p, dict):
        return float(p.get("x", 0.0)), float(p.get("y", 0.0))
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    return 0.0, 0.0


def text_block(
    text: str,
    x: float,
    y: float,
    width: float,
    font: float,
    color: str,
    anchor: str = "start",
    vertical: str = "start",
    height: float | None = None,
) -> str:
    """SVG <text> with wrapped lines. `vertical` positions the block inside `height`."""
    lines = wrap_text(text, width, font)
    lh = font * 1.3
    block_h = lh * len(lines)
    top = y
    if height is not None:
        top = y + {"start": 0.0, "middle": (height - block_h) / 2, "end": height - block_h}.get(
            vertical, 0.0
        )
    tx = x + {"start": 0.0, "middle": width / 2, "end": width}.get(anchor, 0.0)
    out = [
        f'<text font-family="Arial, Helvetica, sans-serif" font-size="{fmt(font)}" '
        f'fill="{color}" text-anchor="{anchor}">'
    ]
    for i, line in enumerate(lines):
        out.append(f'<tspan x="{fmt(tx)}" y="{fmt(top + lh * (i + 0.8))}">{escape(line)}</tspan>')
    out.append("</text>")
    return "".join(out)


def _anchor(value: Any) -> str:
    return {"start": "start", "middle": "middle", "end": "end"}.get(str(value), "middle")


# ---------------------------------------------------------------------------- shapes


def _arrow_end(point: Any, visible: dict[str, NormalizedAnnotation], origin: tuple[float, float]):
    """Arrow end in local coordinates. Attached ends use the bound shape's box."""
    if isinstance(point, dict) and point.get("type") == "binding":
        bound = visible.get(str(point.get("boundShapeId", "")))
        if bound is not None:
            anchor = point.get("normalizedAnchor") or {"x": 0.5, "y": 0.5}
            ax = bound.bounds.x + float(anchor.get("x", 0.5)) * bound.bounds.width
            ay = bound.bounds.y + float(anchor.get("y", 0.5)) * bound.bounds.height
            return ax - origin[0], ay - origin[1]
        return None
    return _point(point)


def _arrowhead(tip: tuple[float, float], frm: tuple[float, float], size: float, color: str) -> str:
    angle = math.atan2(tip[1] - frm[1], tip[0] - frm[0])
    a1, a2 = angle + math.radians(155), angle - math.radians(155)
    p1 = (tip[0] + size * math.cos(a1), tip[1] + size * math.sin(a1))
    p2 = (tip[0] + size * math.cos(a2), tip[1] + size * math.sin(a2))
    pts = f"{fmt(p1[0])},{fmt(p1[1])} {fmt(tip[0])},{fmt(tip[1])} {fmt(p2[0])},{fmt(p2[1])}"
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="3" />'


def render_shape(ann: NormalizedAnnotation, visible: dict[str, NormalizedAnnotation]) -> str:
    """One annotation as an SVG group in slide coordinates."""
    st = ann.shape_state
    props = st.get("props") if isinstance(st.get("props"), dict) else {}
    if not st:  # no merged state stored: show where it was
        b = ann.bounds
        return (
            f'<rect x="{fmt(b.x)}" y="{fmt(b.y)}" width="{fmt(b.width)}" height="{fmt(b.height)}" '
            'fill="none" stroke="#999" stroke-dasharray="6 4" />'
        )
    x, y = float(st.get("x", 0.0)), float(st.get("y", 0.0))
    rot = math.degrees(float(st.get("rotation", 0.0) or 0.0))
    opacity = float(st.get("opacity", 1.0) or 1.0)
    color = _color(props)
    width = STROKE.get(str(props.get("size", "m")), 3.5)
    kind = ann.shape_type
    body = ""

    if kind in (ShapeType.DRAW, ShapeType.HIGHLIGHT):
        sw = width * 5 if kind == ShapeType.HIGHLIGHT else width
        alpha = ' stroke-opacity="0.35"' if kind == ShapeType.HIGHLIGHT else ""
        for seg in props.get("segments", []):
            pts = [_point(p) for p in (seg.get("points", []) if isinstance(seg, dict) else [])]
            if len(pts) == 1:
                body += (
                    f'<circle cx="{fmt(pts[0][0])}" cy="{fmt(pts[0][1])}" r="{fmt(sw / 2)}" '
                    f'fill="{color}" />'
                )
            elif pts:
                joined = " ".join(f"{fmt(px)},{fmt(py)}" for px, py in pts)
                body += (
                    f'<polyline points="{joined}" fill="none" stroke="{color}" '
                    f'stroke-width="{fmt(sw)}" stroke-linecap="round" stroke-linejoin="round"'
                    f"{alpha} />"
                )
    elif kind == ShapeType.GEO:
        w, h = float(props.get("w", 0.0)), float(props.get("h", 0.0))
        geo = str(props.get("geo", "rectangle"))
        fill_mode = str(props.get("fill", "none"))
        fill = (
            f'fill="{color}" fill-opacity="{0.35 if fill_mode == "solid" else 0.15}"'
            if fill_mode != "none"
            else 'fill="none"'
        )
        common = f'{fill} stroke="{color}" stroke-width="{fmt(width)}" stroke-linejoin="round"'
        if geo == "ellipse":
            body += (
                f'<ellipse cx="{fmt(w / 2)}" cy="{fmt(h / 2)}" rx="{fmt(w / 2)}" '
                f'ry="{fmt(h / 2)}" {common} />'
            )
        elif geo in ("triangle", "diamond"):
            pts = (
                [(w / 2, 0), (w, h), (0, h)]
                if geo == "triangle"
                else [(w / 2, 0), (w, h / 2), (w / 2, h), (0, h / 2)]
            )
            joined = " ".join(f"{fmt(px)},{fmt(py)}" for px, py in pts)
            body += f'<polygon points="{joined}" {common} />'
        else:
            body += f'<rect width="{fmt(w)}" height="{fmt(h)}" {common} />'
        label = str(props.get("text", "") or "")
        if label:
            font = LABEL_SIZE.get(str(props.get("size", "m")), 22.0)
            body += text_block(
                label,
                16,
                0,
                max(w - 32, 20),
                font,
                _color(props, "labelColor"),
                _anchor(props.get("align", "middle")),
                _anchor(props.get("verticalAlign", "middle")),
                height=h,
            )
    elif kind == ShapeType.NOTE:
        w = float(props.get("w", 200.0) or 200.0)
        base_h = float(props.get("h", 200.0) or 200.0) + float(props.get("growY", 0.0) or 0.0)
        font = LABEL_SIZE.get(str(props.get("size", "m")), 22.0)
        text = str(props.get("text", "") or "")
        needed = len(wrap_text(text, w - 32, font)) * font * 1.3 + 32 if text else 0
        h = max(base_h, needed)
        fill = NOTE_FILL.get(str(props.get("color", "black")), "#fce19c")
        body += (
            f'<rect width="{fmt(w)}" height="{fmt(h)}" rx="4" fill="{fill}" stroke="#00000022" />'
        )
        if text:
            body += text_block(
                text,
                16,
                16,
                w - 32,
                font,
                COLORS["black"],
                _anchor(props.get("align", "start")),
                _anchor(props.get("verticalAlign", "start")),
                height=h - 32,
            )
    elif kind == ShapeType.TEXT:
        text = str(props.get("text", "") or "")
        font = TEXT_SIZE.get(str(props.get("size", "m")), 24.0)
        box_w = float(props.get("w", 0.0) or 0.0)
        body += text_block(
            text,
            0,
            0,
            box_w if box_w > 40 else 700.0,
            font,
            color,
            _anchor(props.get("textAlign", props.get("align", "start"))),
        )
    elif kind == ShapeType.ARROW:
        start = _arrow_end(props.get("start"), visible, (x, y))
        end = _arrow_end(props.get("end"), visible, (x, y))
        if start and end:
            body += (
                f'<line x1="{fmt(start[0])}" y1="{fmt(start[1])}" x2="{fmt(end[0])}" '
                f'y2="{fmt(end[1])}" stroke="{color}" stroke-width="{fmt(width)}" '
                'stroke-linecap="round" />'
            )
            if str(props.get("arrowheadEnd", "arrow")) != "none":
                body += _arrowhead(end, start, 12 + width * 2, color)
            if str(props.get("arrowheadStart", "none")) != "none":
                body += _arrowhead(start, end, 12 + width * 2, color)
            label = str(props.get("text", "") or "")
            if label:
                mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                font = LABEL_SIZE.get(str(props.get("size", "m")), 22.0)
                body += text_block(
                    label,
                    mx - 100,
                    my - font * 1.3,
                    200,
                    font,
                    _color(props, "labelColor"),
                    "middle",
                )
    elif kind == ShapeType.LINE:
        handles = props.get("handles") if isinstance(props.get("handles"), dict) else {}
        pts = [_point(h) for h in sorted(handles.values(), key=lambda v: str(v.get("index", "")))]
        if pts:
            joined = " ".join(f"{fmt(px)},{fmt(py)}" for px, py in pts)
            body += (
                f'<polyline points="{joined}" fill="none" stroke="{color}" '
                f'stroke-width="{fmt(width)}" stroke-linecap="round" stroke-linejoin="round" />'
            )
    elif kind == ShapeType.FRAME:
        w, h = float(props.get("w", 0.0)), float(props.get("h", 0.0))
        body += (
            f'<rect width="{fmt(w)}" height="{fmt(h)}" fill="none" stroke="#8e9ba8" '
            'stroke-width="2" />'
        )
        name = str(props.get("name", "") or "")
        if name:
            body += text_block(name, 0, -30, max(w, 100), 20, "#555555")
    else:
        b = ann.bounds
        body += (
            f'<rect width="{fmt(b.width)}" height="{fmt(b.height)}" fill="none" stroke="#999" '
            'stroke-dasharray="6 4" />'
        )

    transform = f"translate({fmt(x)} {fmt(y)})" + (f" rotate({fmt(rot)})" if rot else "")
    op = f' opacity="{fmt(opacity)}"' if opacity < 1.0 else ""
    return f'<g transform="{transform}"{op}>{body}</g>'


def fullest_time(timeline: AnnotationTimeline) -> float:
    """The latest moment at which the most shapes were visible together on the slide."""
    best_t, best_n = 0.0, -1
    for t in timeline.get_keyframes():
        n = len(timeline.get_active_annotations(t))
        if n >= best_n:
            best_t, best_n = t, n
    return best_t


def render_board(
    view: SlideView, at_seconds: float, size: tuple[float, float]
) -> tuple[str, list[NormalizedAnnotation]]:
    """SVG of the slide with the annotations visible at `at_seconds`."""
    snap = reconstruct_snapshot(view.timeline, at_seconds)
    visible = {a.shape_id: a for a in snap.annotations}
    ordered = sorted(
        snap.annotations,
        key=lambda a: (str(a.shape_state.get("index", "")), a.timestamp_seconds, a.shape_id),
    )
    w, h = size
    parts = [
        f'<svg class="board" viewBox="0 0 {fmt(w)} {fmt(h)}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="Slide with the lecturer\'s annotations">'
    ]
    if view.svg:
        data = base64.b64encode(view.svg.encode("utf-8")).decode("ascii")
        parts.append(
            f'<image href="data:image/svg+xml;base64,{data}" x="0" y="0" '
            f'width="{fmt(w)}" height="{fmt(h)}" />'
        )
    else:
        parts.append(f'<rect width="{fmt(w)}" height="{fmt(h)}" fill="#f4f4f4" />')
    parts.append(f'<rect width="{fmt(w)}" height="{fmt(h)}" fill="none" stroke="#cccccc" />')
    parts.extend(render_shape(a, visible) for a in ordered)
    parts.append("</svg>")
    return "".join(parts), snap.annotations


# ---------------------------------------------------------------------------- page


def _shape_label(ann: NormalizedAnnotation) -> str:
    if ann.shape_type == ShapeType.GEO:
        geo = str((ann.shape_state.get("props") or {}).get("geo", "rectangle"))
        return geo.replace("-", " ").capitalize()
    return TYPE_LABEL.get(ann.shape_type, "Item")


def _timeline_rows(view: SlideView, visible_ids: set[str]) -> str:
    by_shape: dict[str, list[NormalizedAnnotation]] = {}
    clears: list[float] = []
    for ev in view.timeline.events:
        if ev.is_clear:
            clears.append(ev.timestamp_seconds)
        elif not ev.is_deleted:
            by_shape.setdefault(ev.shape_id, []).append(ev)
    rows: list[tuple[float, str]] = []
    for shape_id, versions in by_shape.items():
        first, last = versions[0], versions[-1]
        text = next((v.text_content for v in reversed(versions) if v.text_content), None)
        note = []
        edits = len(versions) - 1
        if edits > 0:
            note.append(f"edited {edits} time{'s' if edits != 1 else ''}")
        undo = [v.timestamp_undo for v in versions if v.timestamp_undo is not None]
        if shape_id not in visible_ids and undo:
            note.append(f"removed at {clock(min(undo))}")
        elif shape_id not in visible_ids:
            note.append("not visible at the chosen time")
        quote = f"<q>{escape(text)}</q>" if text else ""
        extra = f' <span class="muted">({escape(", ".join(note))})</span>' if note else ""
        rows.append(
            (
                first.timestamp_seconds,
                f'<li><span class="t">{clock(first.timestamp_seconds)}</span> '
                f"<b>{escape(_shape_label(last))}</b> {quote}{extra}</li>",
            )
        )
    for t in clears:
        rows.append((t, f'<li><span class="t">{clock(t)}</span> <b>Board cleared</b></li>'))
    rows.sort(key=lambda r: r[0])
    return '<ol class="timeline">' + "".join(r[1] for r in rows) + "</ol>" if rows else ""


def _groups_rows(view: SlideView) -> str:
    if not view.groups:
        return ""
    items = []
    for g in view.groups:
        label = TYPE_LABEL.get(g.primary_type, "Item")
        text = f" <q>{escape(g.native_text)}</q>" if g.native_text else ""
        items.append(
            f'<li><span class="t">{clock(g.start_seconds)}&ndash;{clock(g.end_seconds)}</span> '
            f"{g.member_count} item{'s' if g.member_count != 1 else ''}, mostly "
            f"{escape(label.lower())}{text}</li>"
        )
    return '<ul class="groups">' + "".join(items) + "</ul>"


def _diagram_rows(diagrams: list[Diagram]) -> str:
    items = []
    for d in diagrams:
        lines = "".join(f"<div>{escape(s)}</div>" for s in d.sentences)
        stamp = f'<span class="t">{clock(d.first_seconds)}</span>'
        items.append(f'<li>{stamp}<span class="dg">{lines}</span></li>')
    return '<ul class="groups">' + "".join(items) + "</ul>" if items else ""


def _slide_title(view: SlideView) -> str:
    number = view.info.number if view.info and view.info.number is not None else None
    heading = f"Slide {number}" if number is not None else "Slide"
    first_line = ""
    if view.info and view.info.text:
        first_line = next((ln.strip() for ln in view.info.text.splitlines() if ln.strip()), "")
    return heading + (f" &mdash; {escape(first_line[:90])}" if first_line else "")


STYLE = """
body{font-family:Arial,Helvetica,sans-serif;color:#1d1d1d;margin:0;background:#f3f4f6}
main{max-width:1100px;margin:0 auto;padding:24px}
h1{margin:0 0 4px;font-size:28px}h2{font-size:20px;margin:0 0 12px}
h3{font-size:15px;margin:18px 0 6px}
.muted{color:#6b7280}.summary{margin:0 0 24px}
section.slide{background:#fff;border-radius:8px;padding:20px;margin:0 0 24px;
  box-shadow:0 1px 3px rgba(0,0,0,.12)}
svg.board{width:100%;height:auto;display:block;border:1px solid #d1d5db}
.timeline,.groups{padding-left:0;list-style:none;margin:0}
.timeline li,.groups li{padding:6px 0;border-bottom:1px solid #eee;line-height:1.4}
.t{display:inline-block;min-width:64px;font-variant-numeric:tabular-nums;color:#374151;
  font-weight:bold}
.dg{display:inline-block;vertical-align:top}
figure{margin:0 0 14px}figcaption{font-size:13px;color:#374151;margin:0 0 4px}
q{color:#111}details{margin-top:12px}summary{cursor:pointer;color:#374151}
pre{white-space:pre-wrap;background:#f9fafb;padding:10px;border-radius:6px;font-size:13px}
footer{color:#6b7280;font-size:12px;margin-top:12px}
@page{size:A4;margin:12mm}
@media print{body{background:#fff}main{padding:0;max-width:none}
  section.slide{box-shadow:none;break-inside:avoid-page;padding:0;margin:0 0 14mm}}
"""


def build_report_html(
    lecture_id: str,
    views: list[SlideView],
    at_seconds: float | None = None,
    generated: datetime | None = None,
) -> str:
    """The whole report page. Without `at_seconds` each slide shows its fullest board (and the
    final board when that differs); with it, the board at that second."""
    all_events = [e for v in views for e in v.timeline.events]
    times = [e.timestamp_seconds for e in all_events]
    first, last = (min(times), max(times)) if times else (0.0, 0.0)
    count = sum(1 for e in all_events if not e.is_deleted and not e.is_clear)
    stamp = (generated or datetime.now(UTC)).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []
    for view in views:
        size = DEFAULT_SIZE
        if view.info and view.info.scaled_width and view.info.scaled_height:
            size = (view.info.scaled_width, view.info.scaled_height)
        end_t = max((e.timestamp_seconds for e in view.timeline.events), default=0.0)
        if at_seconds is not None:
            moments = [(f"Board at {clock(at_seconds)}", at_seconds)]
        else:
            fullest = fullest_time(view.timeline)
            moments = [(f"Board at its fullest ({clock(fullest)})", fullest)]
            ids_full = {a.shape_id for a in view.timeline.get_active_annotations(fullest)}
            ids_end = {a.shape_id for a in view.timeline.get_active_annotations(end_t)}
            if ids_end != ids_full:
                moments.append((f"Board at the end ({clock(end_t)})", end_t))
        figures = []
        visible_ids: set[str] = set()
        diagrams: dict[str, Diagram] = {}
        for caption, moment in moments:
            svg_board, visible = render_board(view, moment, size)
            visible_ids |= {a.shape_id for a in visible}
            for d in build_diagrams(visible):
                diagrams.setdefault(d.diagram_id, d)
            figures.append(
                f"<figure><figcaption>{escape(caption)} &middot; {len(visible)} items"
                f"</figcaption>{svg_board}</figure>"
            )
        slide_text = ""
        if view.info and view.info.text and view.info.text.strip():
            slide_text = (
                f"<details><summary>Slide text</summary><pre>{escape(view.info.text.strip())}"
                "</pre></details>"
            )
        sections.append(
            f'<section class="slide"><h2>{_slide_title(view)}</h2>{"".join(figures)}'
            f"{'<h3>Diagrams</h3>' + _diagram_rows(list(diagrams.values())) if diagrams else ''}"
            f"<h3>What was written and drawn</h3>{_timeline_rows(view, visible_ids)}"
            f"{'<h3>Grouped items</h3>' + _groups_rows(view) if view.groups else ''}"
            f"{slide_text}</section>"
        )

    body = (
        "".join(sections) or '<p class="muted">No annotations were captured for this lecture.</p>'
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "img-src data:; style-src 'unsafe-inline'\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Lecture report</title><style>{STYLE}</style></head><body><main>"
        f'<h1>Lecture report</h1><p class="summary muted">Meeting {escape(lecture_id)}. '
        f"{len(views)} slide{'s' if len(views) != 1 else ''} with annotations, {count} items "
        f"written or drawn, between {clock(first)} and {clock(last)} after the meeting started."
        f"</p>{body}<footer>Generated {escape(stamp)}. Times are measured from the start of the "
        "meeting. Drawings are shown where the lecturer placed them on each slide.</footer>"
        "</main></body></html>"
    )
