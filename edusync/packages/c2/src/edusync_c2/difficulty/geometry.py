"""Pure slide-geometry helpers for the materials parser (integer EMU, no python-pptx imports).

A *rect* is ``(x, y, width, height)`` as stored in a PPTX; a *box* is ``(x0, y0, x1, y1)``.
Everything is integer arithmetic so the same deck always yields the same areas.
"""

from dataclasses import dataclass
from itertools import pairwise

# Above this many boxes on one slide the exact union (quadratic coordinate compression) is
# replaced by the sum of the clamped boxes, itself clamped to the slide area. That fallback can
# overestimate overlap-heavy slides, but is bounded and cheap.
MAX_UNION_SHAPES = 200

Rect = tuple[int, int, int, int]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class GroupFrame:
    """A group's mapping from child space (chOff/chExt) to its parent space (off/ext)."""

    off_x: int
    off_y: int
    ext_cx: int
    ext_cy: int
    ch_off_x: int
    ch_off_y: int
    ch_ext_cx: int
    ch_ext_cy: int


def apply_rotation(rect: Rect, degrees: float) -> Rect:
    """Swap width and height about the centre for rotations of 45-135 and 225-315 degrees."""
    x, y, w, h = rect
    angle = degrees % 360
    if 45 <= angle < 135 or 225 <= angle < 315:
        return ((2 * x + w - h) // 2, (2 * y + h - w) // 2, h, w)
    return rect


def map_through_group(rect: Rect, frame: GroupFrame) -> Rect:
    """Map a child-space rect into the group's parent space.

    Raises ``ValueError`` when the group's child extent is zero (the mapping is undefined).
    """
    if frame.ch_ext_cx == 0 or frame.ch_ext_cy == 0:
        raise ValueError("group child extent is zero")
    x, y, w, h = rect
    return (
        frame.off_x + (x - frame.ch_off_x) * frame.ext_cx // frame.ch_ext_cx,
        frame.off_y + (y - frame.ch_off_y) * frame.ext_cy // frame.ch_ext_cy,
        w * frame.ext_cx // frame.ch_ext_cx,
        h * frame.ext_cy // frame.ch_ext_cy,
    )


def to_slide_space(rect: Rect, frames: list[GroupFrame]) -> Rect:
    """Map a rect through nested groups, innermost first (``frames`` is outermost first)."""
    for frame in reversed(frames):
        rect = map_through_group(rect, frame)
    return rect


def to_box(rect: Rect) -> Box:
    x, y, w, h = rect
    return (x, y, x + w, y + h)


def clamp_box(box: Box, slide_w: int, slide_h: int) -> Box | None:
    """Clip a box to the slide; ``None`` when nothing remains (off-slide or zero size)."""
    x0, y0, x1, y1 = box
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, slide_w), min(y1, slide_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def union_area(boxes: list[Box], slide_w: int, slide_h: int) -> int:
    """Area (EMU squared) covered by the union of the boxes, clipped to the slide."""
    clipped = [b for b in (clamp_box(box, slide_w, slide_h) for box in boxes) if b is not None]
    if len(clipped) > MAX_UNION_SHAPES:
        total = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in clipped)
        return min(total, slide_w * slide_h)
    xs = sorted({x for b in clipped for x in (b[0], b[2])})
    area = 0
    for left, right in pairwise(xs):
        spans = sorted((b[1], b[3]) for b in clipped if b[0] <= left and right <= b[2])
        covered = 0
        current_end = None
        for start, end in spans:
            if current_end is None or start > current_end:
                covered += end - start
                current_end = end
            elif end > current_end:
                covered += end - current_end
                current_end = end
        area += covered * (right - left)
    return area
