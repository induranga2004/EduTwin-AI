"""C2: learning difficulty prediction and comprehension validation (heuristic baselines)."""

from edusync_c2.router import router
from edusync_core import Component

COMPONENT = Component(
    name="c2",
    title="Learning difficulty and comprehension",
    description=(
        "Pre-lecture difficulty prediction pipeline (rule-based baseline, mocked C1 graph) "
        "and poll-based comprehension flags."
    ),
    router=router,
)

__all__ = ["COMPONENT", "router"]
