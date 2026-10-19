"""EduSync C1: lecturer monitoring (document concepts, transcription, coverage timeline)."""

from edusync_c1.api import router
from edusync_core.component import Component

COMPONENT = Component(
    name="c1",
    title="Lecturer Monitoring",
    description=(
        "Lecture document processing, speech transcription, candidate knowledge graph "
        "and timestamped concept-coverage timeline."
    ),
    router=router,
)

__all__ = ["COMPONENT", "router"]
