"""Normalization and geometry extraction for BigBlueButton / tldraw whiteboard events."""

import copy
import hashlib
import json
import math
from datetime import datetime
from typing import Any, Literal

from edusync_c4.annotations import parse_annotation_text
from edusync_c4.schemas import BoundingBox, NormalizedAnnotation, ShapeType, StrokePoint


def apply_delta(current_state: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    """Recursively deep-merge an incoming delta dictionary into a copy of current_state."""
    merged = copy.deepcopy(current_state)
    for key, val in delta.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = apply_delta(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def compute_bounds(
    shape_type: ShapeType | str,
    props: dict[str, Any],
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> BoundingBox:
    """Compute an axis-aligned bounding box from shape properties and canvas coordinates."""
    stype = str(shape_type).lower()
    if isinstance(shape_type, ShapeType):
        stype = shape_type.value

    # Freehand stroke or highlight
    if stype in (ShapeType.DRAW.value, ShapeType.HIGHLIGHT.value, "draw", "highlight"):
        xs: list[float] = []
        ys: list[float] = []
        for segment in props.get("segments", []):
            points = segment.get("points") if isinstance(segment, dict) else None
            if not isinstance(points, list):
                continue
            for pt in points:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    xs.append(origin_x + float(pt[0]))
                    ys.append(origin_y + float(pt[1]))
                elif isinstance(pt, dict) and "x" in pt and "y" in pt:
                    xs.append(origin_x + float(pt["x"]))
                    ys.append(origin_y + float(pt["y"]))
        if xs and ys:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            return BoundingBox(
                x=round(min_x, 2),
                y=round(min_y, 2),
                width=round(max(0.0, max_x - min_x), 2),
                height=round(max(0.0, max_y - min_y), 2),
            )

    # Arrow shape
    if stype in (ShapeType.ARROW.value, "arrow"):
        start = props.get("start")
        end = props.get("end")
        if isinstance(start, dict) and isinstance(end, dict):
            sx = origin_x + float(start.get("x", 0.0))
            sy = origin_y + float(start.get("y", 0.0))
            ex = origin_x + float(end.get("x", 0.0))
            ey = origin_y + float(end.get("y", 0.0))
            min_x, max_x = min(sx, ex), max(sx, ex)
            min_y, max_y = min(sy, ey), max(sy, ey)
            return BoundingBox(
                x=round(min_x, 2),
                y=round(min_y, 2),
                width=round(max(0.0, max_x - min_x), 2),
                height=round(max(0.0, max_y - min_y), 2),
            )

    # Line shape
    if stype in (ShapeType.LINE.value, "line"):
        handles = props.get("handles")
        if isinstance(handles, dict):
            pts: list[tuple[float, float]] = []
            for h in handles.values():
                if isinstance(h, dict) and "x" in h and "y" in h:
                    pts.append((origin_x + float(h["x"]), origin_y + float(h["y"])))
            if pts:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                return BoundingBox(
                    x=round(min(xs), 2),
                    y=round(min(ys), 2),
                    width=round(max(0.0, max(xs) - min(xs)), 2),
                    height=round(max(0.0, max(ys) - min(ys)), 2),
                )

    # Sticky Note shape (default 200x200 + growY in tldraw v2)
    if stype in (ShapeType.NOTE.value, "note"):
        w = float(props.get("w", 200.0))
        h = float(props.get("h", 200.0)) + float(props.get("growY", 0.0))
        return BoundingBox(
            x=round(origin_x, 2),
            y=round(origin_y, 2),
            width=round(max(0.0, w), 2),
            height=round(max(0.0, h), 2),
        )

    # Standard rectangular shapes (text, geo, frame, poll)
    w = float(props.get("w", 0.0))
    h = float(props.get("h", 0.0)) + float(props.get("growY", 0.0))
    return BoundingBox(
        x=round(origin_x, 2),
        y=round(origin_y, 2),
        width=round(max(0.0, w), 2),
        height=round(max(0.0, h), 2),
    )


def extract_stroke_points(
    props: dict[str, Any],
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> list[StrokePoint]:
    """Extract and offset all stroke points from freehand segments."""
    points_out: list[StrokePoint] = []
    for segment in props.get("segments", []):
        if not isinstance(segment, dict):
            continue
        pts = segment.get("points")
        if not isinstance(pts, list):
            continue
        for pt in pts:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                pr = float(pt[2]) if len(pt) > 2 else 0.5
                points_out.append(
                    StrokePoint(
                        x=round(origin_x + float(pt[0]), 2),
                        y=round(origin_y + float(pt[1]), 2),
                        pressure=round(pr, 3),
                    )
                )
            elif isinstance(pt, dict) and "x" in pt and "y" in pt:
                pr = float(pt.get("z", pt.get("pressure", 0.5)))
                points_out.append(
                    StrokePoint(
                        x=round(origin_x + float(pt["x"]), 2),
                        y=round(origin_y + float(pt["y"]), 2),
                        pressure=round(pr, 3),
                    )
                )
    return points_out


TimeUnit = Literal["s", "ms"]

# (field, unit). Units are fixed per field instead of guessed from magnitude, because a
# magnitude heuristic mislabels both long lectures (> 10,000 s) and small millisecond values.
# "timestamp" is ambiguous in BBB sources, so its unit is supplied by the caller.
_CLEAR_EVENT_NAMES = ("ClearWhiteboardEvent", "ClearWhiteboardEvtMsg")


def _timestamp_candidates(time_unit: TimeUnit) -> tuple[tuple[str, str], ...]:
    return (
        ("timestamp_seconds", "s"),
        ("timestamp", time_unit),
        ("updatedAt", "ms-epoch"),
        ("lastUpdatedAt", "ms-epoch"),
        ("timestampUTC", "ms-epoch"),
        ("time", "s"),
    )


def resolve_timestamp(
    raw: dict[str, Any],
    time_unit: TimeUnit = "s",
    lecture_start_ms: float | None = None,
) -> float | None:
    """Return the event time in lecture seconds, or None if no usable timestamp exists.

    `updatedAt`, `lastUpdatedAt` and `timestampUTC` are epoch milliseconds; when
    `lecture_start_ms` is given it is subtracted so the result is lecture-relative.
    Negative or non-finite results are rejected rather than clamped.
    """
    for key, unit in _timestamp_candidates(time_unit):
        val = raw.get(key)
        if val is None or isinstance(val, bool):
            continue
        seconds: float | None = None
        if isinstance(val, (int, float)):
            seconds = float(val) / 1000.0 if unit.startswith("ms") else float(val)
        elif isinstance(val, str) and val.strip():
            try:
                fval = float(val)
                seconds = fval / 1000.0 if unit.startswith("ms") else fval
            except ValueError:
                try:
                    seconds = datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
                    unit = "ms-epoch"  # ISO strings are absolute wall-clock time
                except ValueError:
                    continue
        if seconds is None or not math.isfinite(seconds):
            continue
        if unit == "ms-epoch" and lecture_start_ms is not None:
            seconds -= lecture_start_ms / 1000.0
        if seconds < 0:
            continue
        return seconds
    return None


def _digest(raw: dict[str, Any]) -> str:
    """Deterministic short hash of a raw event, used for fallback ids."""
    blob = json.dumps(raw, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _parse_shape_type(raw_type: str | None) -> ShapeType:
    if not raw_type:
        return ShapeType.UNKNOWN
    clean = raw_type.lower().strip()
    for st in ShapeType:
        if st.value == clean:
            return st
    return ShapeType.UNKNOWN


def normalize_event(
    raw: dict[str, Any],
    default_lecture_id: str = "default",
    default_slide_id: str = "default",
    state_cache: dict[str, dict[str, Any]] | None = None,
    time_unit: TimeUnit = "s",
    lecture_start_ms: float | None = None,
) -> NormalizedAnnotation:
    """Normalize a raw BBB/tldraw whiteboard event into a structured NormalizedAnnotation.

    Handles full shapes, delta history entries, deletion events, whiteboard-clear events
    and Redis envelopes. Never raises on malformed input: an event without a usable
    timestamp is returned with `timestamp_valid=False` so callers can keep it for audit.
    """
    # 1. Resolve meeting/lecture ID
    lecture_id = str(
        raw.get("meetingId")
        or raw.get("meeting_id")
        or raw.get("lecture_id")
        or (raw.get("meta", {}).get("meetingId") if isinstance(raw.get("meta"), dict) else None)
        or default_lecture_id
    )

    # 2. Resolve slide/page ID
    meta_dict = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    slide_id = str(
        raw.get("pageId")
        or raw.get("page_id")
        or raw.get("slide_id")
        or raw.get("whiteboardId")
        or meta_dict.get("presentationId")
        or default_slide_id
    )

    # 3. Detect deletion / clear event
    is_clear = bool(raw.get("is_clear") or raw.get("eventName") in _CLEAR_EVENT_NAMES)
    is_deleted = is_clear or bool(
        raw.get("is_deleted")
        or ("annotationInfo" in raw and raw["annotationInfo"] is None)
        or raw.get("annotationInfo") == ""
        or raw.get("type") in ("delete", "undo")
        or raw.get("eventName") in ("DeleteWhiteboardAnnotationsPubMsg", "DeleteTldrawShapeEvent")
    )

    # 4. Extract shape payload
    shape_payload: dict[str, Any] = {}
    info = raw.get("annotationInfo")
    if is_clear:
        shape_payload = {}
    elif isinstance(info, str) and info.strip():
        try:
            shape_payload = json.loads(info)
        except ValueError:
            shape_payload = {}
    elif isinstance(info, dict):
        shape_payload = info
    elif isinstance(raw.get("shape_data"), dict):
        shape_payload = raw["shape_data"]
    elif "props" in raw or "type" in raw:
        shape_payload = raw

    # 5. Resolve shape ID
    if is_clear:
        shape_id = f"clear:{slide_id}"
    else:
        shape_id = str(
            raw.get("annotationId")
            or raw.get("shape_id")
            or raw.get("shapeId")
            or shape_payload.get("id")
            or raw.get("id")
            or f"shape:{_digest(raw)}"
        )

    # 6. Apply stateful delta merging if cache is supplied (keyed per slide)
    if state_cache is not None:
        cache_key = f"{slide_id}|{shape_id}"
        if is_clear:
            for key in [k for k in state_cache if k.startswith(f"{slide_id}|")]:
                del state_cache[key]
        elif is_deleted:
            state_cache.pop(cache_key, None)
        else:
            cached = state_cache.get(cache_key, {})
            if cached and shape_payload:
                shape_payload = apply_delta(cached, shape_payload)
            state_cache[cache_key] = shape_payload

    # 7. Extract geometry and coordinates
    origin_x = float(shape_payload.get("x", raw.get("x", 0.0)))
    origin_y = float(shape_payload.get("y", raw.get("y", 0.0)))
    props = shape_payload.get("props") if isinstance(shape_payload.get("props"), dict) else {}

    raw_shape_type = shape_payload.get("type", raw.get("type"))
    shape_type = _parse_shape_type(raw_shape_type)

    bounds = compute_bounds(shape_type, props, origin_x, origin_y)
    stroke_points = extract_stroke_points(props, origin_x, origin_y)

    # 8. Extract text content
    text_content = parse_annotation_text(shape_payload) or parse_annotation_text(raw)

    # 9. Resolve timestamp and lifespan
    resolved_ts = resolve_timestamp(raw, time_unit, lecture_start_ms)
    timestamp_valid = resolved_ts is not None
    ts_seconds = resolved_ts if resolved_ts is not None else 0.0
    ts_undo = ts_seconds if is_deleted and timestamp_valid else None

    # 10. Resolve author / user ID
    payload_meta = shape_payload.get("meta") if isinstance(shape_payload.get("meta"), dict) else {}
    user_id = str(
        raw.get("userId")
        or raw.get("user_id")
        or payload_meta.get("createdBy")
        or meta_dict.get("createdBy")
        or "unknown"
    )

    # The payload digest keeps ids deterministic (idempotent re-import) yet distinct for
    # different events on the same shape in the same millisecond (e.g. update + delete).
    ts_part = f"{ts_seconds:.3f}" if timestamp_valid else "na"
    event_id = str(
        raw.get("event_id") or f"{lecture_id}:{slide_id}:{shape_id}:{ts_part}:{_digest(raw)}"
    )

    return NormalizedAnnotation(
        event_id=event_id,
        lecture_id=lecture_id,
        slide_id=slide_id,
        shape_id=shape_id,
        shape_type=shape_type,
        timestamp_seconds=ts_seconds,
        timestamp_out=None,
        timestamp_undo=ts_undo,
        bounds=bounds,
        stroke_points=stroke_points,
        text_content=text_content,
        is_deleted=is_deleted,
        is_clear=is_clear,
        timestamp_valid=timestamp_valid,
        user_id=user_id,
        raw_payload=raw,
    )


def normalize_events(
    events: list[dict[str, Any]],
    lecture_id: str = "default",
    slide_id: str = "default",
    time_unit: TimeUnit = "s",
    lecture_start_ms: float | None = None,
) -> list[NormalizedAnnotation]:
    """Normalize raw annotation events, tracking state deltas, and return them in input order.

    Deltas are merged in chronological order (stable for equal timestamps) regardless of the
    order of `events`, so a later version never leaks properties into an earlier one. Events
    without a usable timestamp keep their relative position and are merged first.
    """
    state_cache: dict[str, dict[str, Any]] = {}
    chronological = sorted(
        range(len(events)),
        key=lambda i: resolve_timestamp(events[i], time_unit, lecture_start_ms) or 0.0,
    )
    normalized: list[NormalizedAnnotation | None] = [None] * len(events)
    for i in chronological:
        normalized[i] = normalize_event(
            events[i],
            default_lecture_id=lecture_id,
            default_slide_id=slide_id,
            state_cache=state_cache,
            time_unit=time_unit,
            lecture_start_ms=lecture_start_ms,
        )
    return [n for n in normalized if n is not None]
