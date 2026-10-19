"""C4: live visual evidence (slides, whiteboard) and reconciliation with speech evidence."""

from edusync_c4.annotations import parse_annotation_text
from edusync_c4.evidence import merge_evidence, reconcile
from edusync_c4.router import router
from edusync_core import Component

COMPONENT = Component(
    name="c4",
    title="Visual evidence and reconciliation",
    description="Merges slide/whiteboard/speech evidence and flags visual-only concepts.",
    router=router,
)

__all__ = ["COMPONENT", "merge_evidence", "parse_annotation_text", "reconcile", "router"]
