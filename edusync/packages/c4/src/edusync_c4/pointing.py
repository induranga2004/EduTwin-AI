"""Name the slide text that an underline or a circle points at.

Rules (whiteboard coordinates; word boxes come from the slide PDF):
  * Underline: words whose box is covered horizontally by at least half of its width by the
    stroke, and whose bottom edge is just above the stroke (at most `BELOW_TOLERANCE` below the
    word, and not higher than the word's middle). If several text lines qualify, the line closest
    to the stroke wins, so a stroke under one line never names the line above it.
  * Circle or loop: words whose centre lies inside the stroke's outline (winding rule) and which
    lie mostly inside the stroke's bounding box.
Words are returned in reading order. Nothing is guessed: with no match the mark is reported as a
mark with no slide text.

Limits: it needs word boxes (a slide that is only a picture has none); a stroke that cuts through
the middle of text is treated as an underline only if it passes the rules above; it says WHAT was
marked, never why.
"""

from edusync_c4.schemas import Marking, NormalizedAnnotation, SlideWord, StrokeKind, StrokePoint
from edusync_c4.strokes import classify_stroke

BELOW_TOLERANCE = 40.0  # an underline may sit this far under the word's bottom edge
COVER_RATIO = 0.5  # share of a word's width an underline must cover
INSIDE_RATIO = 0.6  # share of a word's box that must lie inside a circle's bounding box
LINE_TOLERANCE = 0.6  # words belong to one line if their vertical centres are this close (x height)


def _readable(words: list[SlideWord]) -> list[SlideWord]:
    """Drop bullet glyphs and other tokens with no letter or digit."""
    return [w for w in words if any(ch.isalnum() for ch in w.text)]


def _reading_order(words: list[SlideWord]) -> list[SlideWord]:
    lines: list[list[SlideWord]] = []
    for w in sorted(words, key=lambda w: (w.y + w.height / 2, w.x)):
        centre = w.y + w.height / 2
        for line in lines:
            ref = line[0]
            if abs((ref.y + ref.height / 2) - centre) <= LINE_TOLERANCE * max(ref.height, w.height):
                line.append(w)
                break
        else:
            lines.append([w])
    return [w for line in lines for w in sorted(line, key=lambda w: w.x)]


def _inside(points: list[StrokePoint], x: float, y: float) -> bool:
    """Is (x, y) inside the outline through the points? (non-zero winding rule)

    The even-odd rule would call the middle of a circle drawn TWICE "outside"; the winding
    number counts repeated loops in the same direction as inside.
    """
    winding = 0
    for p, q in zip(points, [*points[1:], points[0]], strict=True):
        left = (q.x - p.x) * (y - p.y) - (x - p.x) * (q.y - p.y)
        if p.y <= y < q.y and left > 0:
            winding += 1
        elif q.y <= y < p.y and left < 0:
            winding -= 1
    return winding != 0


def underlined_words(points: list[StrokePoint], words: list[SlideWord]) -> list[SlideWord]:
    x0, x1 = min(p.x for p in points), max(p.x for p in points)
    y_line = sum(p.y for p in points) / len(points)
    candidates: list[tuple[float, SlideWord]] = []
    for w in _readable(words):
        covered = max(0.0, min(x1, w.x + w.width) - max(x0, w.x))
        if w.width <= 0 or covered < COVER_RATIO * w.width:
            continue
        bottom = w.y + w.height
        if w.y + 0.5 * w.height <= y_line <= bottom + BELOW_TOLERANCE:
            candidates.append((abs(y_line - bottom), w))
    if not candidates:
        return []
    nearest = min(candidates, key=lambda c: c[0])[1]  # the closest line only
    same_line = [
        w
        for _, w in candidates
        if abs((w.y + w.height) - (nearest.y + nearest.height)) <= LINE_TOLERANCE * nearest.height
    ]
    return _reading_order(same_line)


def circled_words(points: list[StrokePoint], words: list[SlideWord]) -> list[SlideWord]:
    bx0, bx1 = min(p.x for p in points), max(p.x for p in points)
    by0, by1 = min(p.y for p in points), max(p.y for p in points)
    hits: list[SlideWord] = []
    for w in _readable(words):
        if not _inside(points, w.x + w.width / 2, w.y + w.height / 2):
            continue
        ow = max(0.0, min(bx1, w.x + w.width) - max(bx0, w.x))
        oh = max(0.0, min(by1, w.y + w.height) - max(by0, w.y))
        area = w.width * w.height
        if area > 0 and (ow * oh) / area >= INSIDE_RATIO:
            hits.append(w)
    return _reading_order(hits)


def _phrase(words: list[SlideWord]) -> str:
    return " ".join(w.text for w in words)


def describe_marking(ann: NormalizedAnnotation, words: list[SlideWord] | None) -> Marking | None:
    """Classify a freehand annotation and say what it points at. None for non-freehand shapes."""
    if ann.shape_type.value not in ("draw", "highlight"):
        return None
    kind, measures = classify_stroke(ann.stroke_points)
    action = target = None
    if words:
        found: list[SlideWord] = []
        if kind == StrokeKind.UNDERLINE:
            found, label = underlined_words(ann.stroke_points, words), "underlined"
        elif kind in (StrokeKind.CIRCLE, StrokeKind.LOOP):
            label = "circled" if kind == StrokeKind.CIRCLE else "boxed"
            found = circled_words(ann.stroke_points, words)
        if found:
            action, target = label, _phrase(found)
    return Marking(
        shape_id=ann.shape_id,
        kind=kind,
        action=action,
        target_text=target,
        measurements=measures,
    )
