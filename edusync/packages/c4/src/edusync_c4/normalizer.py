"""Normalization and geometry extraction for BigBlueButton / tldraw whiteboard events."""

import copy
import json
import uuid
from datetime import datetime
from typing import Any

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


def _resolve_timestamp(raw: dict[str, Any]) -> float:
    """Extract and normalize timestamp to elapsed/wall-clock seconds."""
    for key in ("timestamp_seconds", "timestamp", "updatedAt", "lastUpdatedAt", "time"):
        val = raw.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            fval = float(val)
            # If timestamp is in milliseconds (> 10,000s), convert to seconds
            return fval / 1000.0 if fval > 10000.0 else fval
        if isinstance(val, str) and val.strip():
            # Try ISO 8601 parsing
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                return float(dt.timestamp())
            except ValueError:
                try:
                    fval = float(val)
                    return fval / 1000.0 if fval > 10000.0 else fval
                except ValueError:
                    continue
    return 0.0


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
) -> NormalizedAnnotation:
    """Normalize a raw BBB/tldraw whiteboard event into a structured NormalizedAnnotation.

    Handles full shapes, delta history entries, deletion events, and Redis envelopes.
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
    is_deleted = bool(
        raw.get("is_deleted")
        or ("annotationInfo" in raw and raw["annotationInfo"] is None)
        or raw.get("annotationInfo") == ""
        or raw.get("type") in ("delete", "undo")
        or raw.get("eventName") in ("DeleteWhiteboardAnnotationsPubMsg", "DeleteTldrawShapeEvent")
    )

    # 4. Extract shape payload
    shape_payload: dict[str, Any] = {}
    info = raw.get("annotationInfo")
    if isinstance(info, str) and info.strip():
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
    shape_id = str(
        raw.get("annotationId")
        or raw.get("shape_id")
        or raw.get("shapeId")
        or shape_payload.get("id")
        or raw.get("id")
        or f"shape:{uuid.uuid4().hex[:8]}"
    )

    # 6. Apply stateful delta merging if cache is supplied
    if state_cache is not None:
        if is_deleted:
            state_cache.pop(shape_id, None)
        else:
            cached = state_cache.get(shape_id, {})
            if cached and shape_payload:
                shape_payload = apply_delta(cached, shape_payload)
            state_cache[shape_id] = shape_payload

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
    ts_seconds = _resolve_timestamp(raw)
    ts_undo = ts_seconds if is_deleted else None

    # 10. Resolve author / user ID
    payload_meta = shape_payload.get("meta") if isinstance(shape_payload.get("meta"), dict) else {}
    user_id = str(
        raw.get("userId")
        or raw.get("user_id")
        or payload_meta.get("createdBy")
        or meta_dict.get("createdBy")
        or "unknown"
    )

    event_id = str(raw.get("event_id") or f"{lecture_id}:{slide_id}:{shape_id}:{ts_seconds:.3f}")

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
        user_id=user_id,
        raw_payload=raw,
    )


def normalize_events(
    events: list[dict[str, Any]],
    lecture_id: str = "default",
    slide_id: str = "default",
) -> list[NormalizedAnnotation]:
    """Normalize a sequence of raw annotation events in temporal order, tracking state deltas."""
    state_cache: dict[str, dict[str, Any]] = {}
    normalized: list[NormalizedAnnotation] = []
    for raw in events:
        norm = normalize_event(
            raw,
            default_lecture_id=lecture_id,
            default_slide_id=slide_id,
            state_cache=state_cache,
        )
        normalized.append(norm)
    return normalized
