"""Rule-based candidate grouping of annotations on a slide.

`SpatioTemporalClusterer` is a deterministic baseline: two shapes are linked when their
bounding boxes are within `max_distance` AND their active time spans are within
`max_gap_seconds`; groups are the connected components of that link graph (single linkage).

Known limitations (by design, not bugs):
  * Single linkage chains: A~B and B~C put A and C together even if far apart.
  * Distance uses axis-aligned bounding boxes; a large freehand loop or a long arrow
    "covers" empty space and can bridge unrelated marks.
  * Thresholds are in virtual slide units and seconds; they are not scaled to slide size or
    to a presenter's pace, so they need tuning per deployment.
  * Groups say nothing about meaning. Equations, diagrams and handwriting are NOT recognised.
  * A shape's time span runs from its first version to its last edit (or retraction); a live,
    never-edited shape therefore has a zero-length span at its creation time.
  * Events without a usable timestamp (`AnnotationTimeline.unplaced`) cannot be placed in time
    and are NOT grouped; they remain available on the timeline for audit.
  * Pairwise comparison is O(n^2) in the number of shapes on one slide.
Replaceable via the `Clusterer` protocol (e.g. a learned segmenter later).
"""

import hashlib
import math
from collections import Counter
from typing import Protocol

from edusync_c4.schemas import (
    AnnotationGroup,
    AnnotationTimeline,
    BoundingBox,
    GroupingConfig,
    NormalizedAnnotation,
    ShapeType,
    SlideSnapshot,
)

METHOD = "spatiotemporal-single-linkage-v1"


class Clusterer(Protocol):
    def group(
        self, timeline: AnnotationTimeline, config: GroupingConfig | None = None
    ) -> list[AnnotationGroup]: ...


def _union_bounds(boxes: list[BoundingBox]) -> BoundingBox:
    x0 = min(b.x for b in boxes)
    y0 = min(b.y for b in boxes)
    x1 = max(b.x + b.width for b in boxes)
    y1 = max(b.y + b.height for b in boxes)
    return BoundingBox(
        x=round(x0, 2), y=round(y0, 2), width=round(x1 - x0, 2), height=round(y1 - y0, 2)
    )


def box_distance(a: BoundingBox, b: BoundingBox) -> float:
    """Gap between two axis-aligned boxes (0 if they touch or overlap)."""
    dx = max(0.0, a.x - (b.x + b.width), b.x - (a.x + a.width))
    dy = max(0.0, a.y - (b.y + b.height), b.y - (a.y + a.height))
    return math.hypot(dx, dy)


def time_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Gap between two [start, end] spans (0 if they overlap)."""
    return max(0.0, max(a[0], b[0]) - min(a[1], b[1]))


class _Shape:
    """All versions of one shape, collapsed to a union box and a time span."""

    def __init__(self, versions: list[NormalizedAnnotation]) -> None:
        self.versions = versions
        self.shape_id = versions[0].shape_id
        self.bounds = _union_bounds([v.bounds for v in versions])
        self.start = min(v.timestamp_seconds for v in versions)
        undos = [v.timestamp_undo for v in versions if v.timestamp_undo is not None]
        self.retracted = any(v.timestamp_undo is not None for v in versions)
        last = max(v.timestamp_seconds for v in versions)
        self.end = max([last, *undos]) if undos else last
        self.shape_type = versions[-1].shape_type


class SpatioTemporalClusterer:
    """Deterministic single-linkage clustering on bounding-box distance and time gap."""

    def group(
        self, timeline: AnnotationTimeline, config: GroupingConfig | None = None
    ) -> list[AnnotationGroup]:
        cfg = config or GroupingConfig()

        by_shape: dict[str, list[NormalizedAnnotation]] = {}
        for ev in timeline.events:
            if ev.is_deleted or ev.is_clear:
                continue
            by_shape.setdefault(ev.shape_id, []).append(ev)
        shapes = sorted(
            (_Shape(vs) for vs in by_shape.values()),
            key=lambda s: (s.start, s.shape_id),
        )
        if not cfg.include_retracted:
            shapes = [s for s in shapes if not s.retracted]

        parent = list(range(len(shapes)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(shapes):
            for j in range(i + 1, len(shapes)):
                b = shapes[j]
                if (
                    box_distance(a.bounds, b.bounds) <= cfg.max_distance
                    and time_gap((a.start, a.end), (b.start, b.end)) <= cfg.max_gap_seconds
                ):
                    parent[find(j)] = find(i)

        components: dict[int, list[_Shape]] = {}
        for i, s in enumerate(shapes):
            components.setdefault(find(i), []).append(s)

        groups = [self._build(timeline, members) for members in components.values()]
        groups.sort(key=lambda g: (g.start_seconds, g.group_id))
        return groups

    @staticmethod
    def _build(timeline: AnnotationTimeline, members: list[_Shape]) -> AnnotationGroup:
        shape_ids = sorted(s.shape_id for s in members)
        versions = [v for s in members for v in s.versions]
        versions.sort(key=lambda v: (v.timestamp_seconds, v.event_id))
        types = Counter(s.shape_type for s in members)
        top = max(types.values())
        primary: ShapeType = sorted((t for t, n in types.items() if n == top), key=str)[0]
        texts: list[str] = []
        for v in versions:
            if v.text_content and v.text_content not in texts:
                texts.append(v.text_content)
        digest = hashlib.sha1("|".join(shape_ids).encode("utf-8")).hexdigest()[:10]
        return AnnotationGroup(
            group_id=f"grp:{timeline.slide_id}:{digest}",
            lecture_id=timeline.lecture_id,
            slide_id=timeline.slide_id,
            method=METHOD,
            shape_ids=shape_ids,
            event_ids=[v.event_id for v in versions],
            retracted_shape_ids=sorted(s.shape_id for s in members if s.retracted),
            member_count=len(members),
            primary_type=primary,
            bounds=_union_bounds([s.bounds for s in members]),
            start_seconds=min(s.start for s in members),
            end_seconds=max(s.end for s in members),
            native_text=" / ".join(texts) if texts else None,
        )


def group_timeline(
    timeline: AnnotationTimeline,
    config: GroupingConfig | None = None,
    clusterer: Clusterer | None = None,
) -> list[AnnotationGroup]:
    """Group one slide's annotations into candidate groups (default: spatio-temporal)."""
    return (clusterer or SpatioTemporalClusterer()).group(timeline, config)


def group_snapshot(
    snapshot: SlideSnapshot,
    config: GroupingConfig | None = None,
    clusterer: Clusterer | None = None,
) -> list[AnnotationGroup]:
    """Group only what is on the board at `snapshot.timestamp_seconds`.

    Future retractions are hidden (`timestamp_out`/`timestamp_undo` cleared on copies), so the
    result depends only on what was visible at that moment, not on later events.
    """
    visible = []
    for ann in snapshot.annotations:
        copy = ann.model_copy(deep=True)
        copy.timestamp_out = None
        copy.timestamp_undo = None
        visible.append(copy)
    timeline = AnnotationTimeline(
        lecture_id=snapshot.lecture_id, slide_id=snapshot.slide_id, events=visible
    )
    return group_timeline(timeline, config, clusterer)
