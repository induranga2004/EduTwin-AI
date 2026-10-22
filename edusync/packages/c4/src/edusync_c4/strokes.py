"""Classify one freehand stroke as an underline, circle, line, dot or scribble.

Plain geometry on the stroke's points, no machine learning; every result carries the numbers it
was based on so it can be shown as evidence. The thresholds are named constants below, tuned on
real strokes from a BigBlueButton lecture (an underline: straightness 0.98, flat; a double-looped
circle: ends 12% of its size apart) plus the synthetic cases in the tests.

Limits: strokes drawn with several separate pen-downs are separate strokes here; a very small
mark is a dot; a "C" that does not nearly close is not a circle; no attempt is made to read
letters or symbols.
"""

import itertools
import math

from edusync_c4.schemas import StrokeKind, StrokePoint

MIN_POINTS = 3
MIN_LENGTH = 20.0  # shorter paths are dots (whiteboard units)
STRAIGHT = 0.85  # chord / path length at or above this: a straight stroke
HORIZONTAL_DEGREES = 12.0  # an underline may tilt this much
UNDERLINE_FLATNESS = 0.20  # height / width at or below this
CLOSED = 0.35  # end-to-start distance / larger bbox side at or below this: closed loop
MIN_LOOP_SIDE = 20.0
MIN_LOOP_LENGTH_RATIO = 1.6  # path length / larger bbox side
MIN_FILL = 0.30  # enclosed area / bbox area, to reject back-and-forth scribbles
CORNER_ZONE = 0.15  # a corner zone is this share of the bbox side, on each axis
BOXY_CORNER_SHARE = 0.18  # at least this share of points in corner zones: a boxy loop


def _path_length(points: list[StrokePoint]) -> float:
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in itertools.pairwise(points))


def polygon_area(points: list[StrokePoint]) -> float:
    """Absolute area of the polygon through the points (shoelace formula)."""
    total = 0.0
    for a, b in zip(points, [*points[1:], points[0]], strict=True):
        total += a.x * b.y - b.x * a.y
    return abs(total) / 2.0


def _corner_share(points: list[StrokePoint], x0: float, y0: float, w: float, h: float) -> float:
    zone_x, zone_y = CORNER_ZONE * w, CORNER_ZONE * h
    near = 0
    for p in points:
        in_x = p.x - x0 <= zone_x or x0 + w - p.x <= zone_x
        in_y = p.y - y0 <= zone_y or y0 + h - p.y <= zone_y
        near += in_x and in_y
    return near / len(points)


def classify_stroke(points: list[StrokePoint]) -> tuple[StrokeKind, dict[str, float]]:
    """(kind, measurements) for one stroke. Measurements are rounded for display."""
    if len(points) < MIN_POINTS:
        return StrokeKind.DOT, {"points": float(len(points))}
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    path = _path_length(points)
    chord = math.hypot(points[-1].x - points[0].x, points[-1].y - points[0].y)
    side = max(width, height)
    measures = {
        "points": float(len(points)),
        "width": round(width, 1),
        "height": round(height, 1),
        "path": round(path, 1),
        "straightness": round(chord / path, 3) if path else 0.0,
        "closure": round(chord / side, 3) if side else 0.0,
    }
    if path < MIN_LENGTH or side < MIN_LENGTH / 2:
        return StrokeKind.DOT, measures

    straightness = chord / path
    if straightness >= STRAIGHT:
        angle = abs(
            math.degrees(math.atan2(points[-1].y - points[0].y, points[-1].x - points[0].x))
        )
        angle = min(angle, 180.0 - angle)  # direction does not matter
        measures["angle"] = round(angle, 1)
        flat = height <= UNDERLINE_FLATNESS * width if width else False
        if angle <= HORIZONTAL_DEGREES and flat:
            return StrokeKind.UNDERLINE, measures
        return StrokeKind.LINE, measures

    closure = chord / side
    if (
        closure <= CLOSED
        and min(width, height) >= MIN_LOOP_SIDE
        and path >= MIN_LOOP_LENGTH_RATIO * side
    ):
        fill = polygon_area(points) / (width * height) if width * height else 0.0
        measures["fill"] = round(fill, 3)
        if fill >= MIN_FILL:
            # A rounded loop has few points near the bbox corners; a drawn box has many. This
            # does not depend on the area, which doubles when the loop is drawn twice.
            share = _corner_share(points, min(xs), min(ys), width, height)
            measures["corner_share"] = round(share, 3)
            return (StrokeKind.LOOP if share >= BOXY_CORNER_SHARE else StrokeKind.CIRCLE), measures
    return StrokeKind.SCRIBBLE, measures
