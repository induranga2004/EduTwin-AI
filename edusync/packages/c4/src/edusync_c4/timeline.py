"""Temporal reconstruction and interval calculation for whiteboard annotations."""

from typing import Any

from edusync_c4.normalizer import TimeUnit, normalize_events
from edusync_c4.schemas import AnnotationTimeline, NormalizedAnnotation, SlideSnapshot


def _to_normalized(
    events: list[NormalizedAnnotation] | list[dict[str, Any]] | list[Any],
    lecture_id: str,
    slide_id: str,
    time_unit: TimeUnit,
    lecture_start_ms: float | None,
) -> list[NormalizedAnnotation]:
    """Normalize raw dicts and deep-copy already-normalized items (inputs are never mutated)."""
    raw_positions = [i for i, ev in enumerate(events) if isinstance(ev, dict)]
    raw_normalized = normalize_events(
        [events[i] for i in raw_positions],
        lecture_id=lecture_id,
        slide_id=slide_id,
        time_unit=time_unit,
        lecture_start_ms=lecture_start_ms,
    )
    by_position = dict(zip(raw_positions, raw_normalized, strict=True))
    result: list[NormalizedAnnotation] = []
    for i, ev in enumerate(events):
        if i in by_position:
            result.append(by_position[i])
        elif isinstance(ev, NormalizedAnnotation):
            result.append(ev.model_copy(deep=True))
        else:
            raise TypeError(f"unsupported event type at index {i}: {type(ev).__name__}")
    return result


def _resolve_intervals(
    norm_events: list[NormalizedAnnotation], lecture_id: str, slide_id: str
) -> AnnotationTimeline:
    """Compute [timestamp_in, timestamp_out, timestamp_undo] for one slide's events.

    Every input event is kept in `events` (or `unplaced` when it has no usable timestamp);
    nothing is dropped. Events at an identical timestamp keep their input order, and a
    version superseded in the same instant simply has an empty visibility window.
    """
    placed = [e for e in norm_events if e.timestamp_valid]
    unplaced = [e for e in norm_events if not e.timestamp_valid]

    # Recompute intervals from scratch so already-resolved inputs give identical results.
    for ev in placed:
        ev.timestamp_out = None
        ev.timestamp_undo = ev.timestamp_seconds if ev.is_deleted else None

    placed.sort(key=lambda e: e.timestamp_seconds)  # stable
    clears = [e for e in placed if e.is_clear]

    by_shape: dict[str, list[NormalizedAnnotation]] = {}
    for ev in placed:
        if not ev.is_clear:
            by_shape.setdefault(ev.shape_id, []).append(ev)

    for shape_events in by_shape.values():
        # `chain` = versions of this shape since its last deletion. Every deletion retracts
        # the whole live chain, so create -> delete -> re-add (undo/redo) -> delete works.
        chain: list[NormalizedAnnotation] = []
        for ev in shape_events:
            if ev.is_deleted:
                for ver in chain:
                    ver.timestamp_undo = ev.timestamp_seconds
                chain = []
                continue
            if chain:
                chain[-1].timestamp_out = ev.timestamp_seconds
            chain.append(ev)

    # A whiteboard clear retracts everything visible on the slide at that instant. When
    # fullClear is explicitly false, only the clearing user's own shapes are retracted.
    for clear in clears:
        t = clear.timestamp_seconds
        full = clear.raw_payload.get("fullClear") is not False
        for ev in placed:
            if ev.is_deleted or ev.timestamp_seconds >= t:
                continue
            if not full and ev.user_id != clear.user_id:
                continue
            if ev.timestamp_out is not None and ev.timestamp_out <= t:
                continue
            if ev.timestamp_undo is not None and ev.timestamp_undo <= t:
                continue
            ev.timestamp_undo = t

    return AnnotationTimeline(
        lecture_id=lecture_id,
        slide_id=slide_id,
        events=placed,
        unplaced=unplaced,
    )


def build_timelines(
    events: list[NormalizedAnnotation] | list[dict[str, Any]] | list[Any],
    lecture_id: str = "default",
    slide_id: str = "default",
    time_unit: TimeUnit = "s",
    lecture_start_ms: float | None = None,
) -> dict[tuple[str, str], AnnotationTimeline]:
    """Build one timeline per (lecture_id, slide_id) found in raw and/or normalized events."""
    norm_events = _to_normalized(events, lecture_id, slide_id, time_unit, lecture_start_ms)
    groups: dict[tuple[str, str], list[NormalizedAnnotation]] = {}
    for ev in norm_events:
        groups.setdefault((ev.lecture_id, ev.slide_id), []).append(ev)
    return {key: _resolve_intervals(group, *key) for key, group in groups.items()}


def build_timeline(
    events: list[NormalizedAnnotation] | list[dict[str, Any]] | list[Any],
    lecture_id: str = "default",
    slide_id: str = "default",
    time_unit: TimeUnit = "s",
    lecture_start_ms: float | None = None,
) -> AnnotationTimeline:
    """Build an interval-resolved annotation timeline for a single slide.

    Calculates:
      - timestamp_in: initial creation / update timestamp.
      - timestamp_out: timestamp when this version was superseded by a later update.
      - timestamp_undo: timestamp when the shape was deleted, undone or cleared.

    Raises ValueError if the events span more than one slide or lecture; use
    `build_timelines` for that. Inputs are never mutated.
    """
    timelines = build_timelines(events, lecture_id, slide_id, time_unit, lecture_start_ms)
    if not timelines:
        return AnnotationTimeline(lecture_id=lecture_id, slide_id=slide_id, events=[])
    if len(timelines) > 1:
        raise ValueError(
            f"events span {len(timelines)} slides/lectures {sorted(timelines)}; "
            "use build_timelines() or filter first"
        )
    return next(iter(timelines.values()))


def reconstruct_snapshot(
    timeline: AnnotationTimeline,
    timestamp_seconds: float,
    slide_text: str | None = None,
) -> SlideSnapshot:
    """Reconstruct the active whiteboard visual state on a slide at timestamp_seconds."""
    active = timeline.get_active_annotations(timestamp_seconds)
    return SlideSnapshot(
        lecture_id=timeline.lecture_id,
        slide_id=timeline.slide_id,
        timestamp_seconds=round(timestamp_seconds, 3),
        annotations=active,
        active_shape_count=len(active),
        slide_text=slide_text,
    )


def replay_timeline(
    timeline: AnnotationTimeline,
    slide_text: str | None = None,
) -> list[SlideSnapshot]:
    """Generate snapshots at every keyframe where visual state changes on the slide."""
    return [
        reconstruct_snapshot(timeline, kf, slide_text=slide_text) for kf in timeline.get_keyframes()
    ]
