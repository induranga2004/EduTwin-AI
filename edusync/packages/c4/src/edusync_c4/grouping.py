"""Rule-based candidate grouping of annotations on a slide.

`SpatioTemporalClusterer` is a deterministic baseline: two shapes are linked when their
bounding boxes are within `max_distance` AND the closest pair of their *authored edit times* is
within `max_gap_seconds`; groups are the connected components of that link graph (single
linkage).

Authored edit times are the timestamps of a shape's versions, EXCLUDING restores. A restore is a
version that reappears after a deletion or a whiteboard clear (BBB's undo re-submits the shape),
so undoing a delete is not treated as the presenter drawing it again. A group's own
`start_seconds`/`end_seconds` still cover the shapes' full lifetimes, including retraction.
A group's `native_text` is the final text of each text-bearing shape, not every typing step.

Known limitations (by design, not bugs):
  * Single linkage chains: A~B and B~C put A and C together even if far apart.
  * Distance uses axis-aligned bounding boxes; a large freehand loop or a long arrow
    "covers" empty space and can bridge unrelated marks.
  * Thresholds are in virtual slide units and seconds; they are not scaled to slide size or
    to a presenter's pace, so they need tuning per deployment.
  * Groups say nothing about meaning. Equations, diagrams and handwriting are NOT recognised.
Replaceable via the `Clusterer` protocol (e.g. a learned segmenter later).
"""

import bisect
import hashlib
import math
from collections import Counter
from typing import Protocol

from edusync_c4.diagram import structural_links
from edusync_c4.schemas import (
    AnnotationGroup,
    AnnotationTimeline,
    BoundingBox,
    GroupingConfig,
    NormalizedAnnotation,
    ShapeType,
)

METHOD = "spatiotemporal-structural-v3"


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


def closest_gap(a: list[float], b: list[float]) -> float:
    """Smallest absolute difference between two sorted, non-empty time lists."""
    best = math.inf
    for t in a:
        i = bisect.bisect_left(b, t)
        if i < len(b):
            best = min(best, b[i] - t)
        if i > 0:
            best = min(best, t - b[i - 1])
    return best


class _Shape:
    """All versions of one shape, collapsed to a union box, a lifetime and its edit times."""

    def __init__(self, versions: list[NormalizedAnnotation], retractions: list[float]) -> None:
        self.versions = versions
        # A version is a restore when a deletion/clear happened since the previous version.
        self.edit_times = sorted(
            v.timestamp_seconds
            for i, v in enumerate(versions)
            if i == 0
            or not any(
                versions[i - 1].timestamp_seconds <= r <= v.timestamp_seconds for r in retractions
            )
        )
        self.final_text = next((v.text_content for v in reversed(versions) if v.text_content), None)
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
        deletions: dict[str, list[float]] = {}
        clears: list[NormalizedAnnotation] = []
        for ev in timeline.events:
            if ev.is_clear:
                clears.append(ev)
            elif ev.is_deleted:
                deletions.setdefault(ev.shape_id, []).append(ev.timestamp_seconds)
            else:
                by_shape.setdefault(ev.shape_id, []).append(ev)

        def retractions_for(versions: list[NormalizedAnnotation]) -> list[float]:
            times = list(deletions.get(versions[0].shape_id, []))
            authors = {v.user_id for v in versions}
            for clear in clears:
                # fullClear=False only removes the clearing user's own shapes
                if clear.raw_payload.get("fullClear") is not False or clear.user_id in authors:
                    times.append(clear.timestamp_seconds)
            return times

        shapes = sorted(
            (_Shape(vs, retractions_for(vs)) for vs in by_shape.values()),
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
                    and closest_gap(a.edit_times, b.edit_times) <= cfg.max_gap_seconds
                ):
                    parent[find(j)] = find(i)

        if cfg.structural_links:
            index = {s.shape_id: i for i, s in enumerate(shapes)}
            latest = [s.versions[-1] for s in shapes]
            for left, right in structural_links(latest):
                if left in index and right in index:
                    parent[find(index[left])] = find(index[right])

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
        for s in sorted(members, key=lambda m: (m.start, m.shape_id)):
            if s.final_text and s.final_text not in texts:
                texts.append(s.final_text)
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
