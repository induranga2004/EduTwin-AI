"""C4: live visual evidence, temporal reconstruction, and speech reconciliation."""

from edusync_c4.annotations import parse_annotation_text
from edusync_c4.evidence import merge_evidence, reconcile
from edusync_c4.normalizer import (
    apply_delta,
    compute_bounds,
    extract_stroke_points,
    normalize_event,
    normalize_events,
)
from edusync_c4.router import router
from edusync_c4.schemas import (
    AnnotationTimeline,
    BoundingBox,
    EventImportRequest,
    EventImportResponse,
    NormalizedAnnotation,
    ReconcileRequest,
    ReconcileResponse,
    ShapeType,
    SlideSnapshot,
    StrokePoint,
)
from edusync_c4.timeline import (
    build_timeline,
    build_timelines,
    reconstruct_snapshot,
    replay_timeline,
)
from edusync_core import Component

COMPONENT = Component(
    name="c4",
    title="Visual evidence and reconciliation",
    description="Merges slide/whiteboard/speech evidence and flags visual-only concepts.",
    router=router,
)

__all__ = [
    "COMPONENT",
    "AnnotationTimeline",
    "BoundingBox",
    "EventImportRequest",
    "EventImportResponse",
    "NormalizedAnnotation",
    "ReconcileRequest",
    "ReconcileResponse",
    "ShapeType",
    "SlideSnapshot",
    "StrokePoint",
    "apply_delta",
    "build_timeline",
    "build_timelines",
    "compute_bounds",
    "extract_stroke_points",
    "merge_evidence",
    "normalize_event",
    "normalize_events",
    "parse_annotation_text",
    "reconcile",
    "reconstruct_snapshot",
    "replay_timeline",
    "router",
]
