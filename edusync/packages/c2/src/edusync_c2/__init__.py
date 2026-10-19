"""C2: learning difficulty prediction and comprehension validation (heuristic baselines)."""

from edusync_c2.router import router
from edusync_core import Component

COMPONENT = Component(
    name="c2",
    title="Learning difficulty and comprehension",
    description="Lexical difficulty heuristic and poll-based comprehension flags per concept.",
    router=router,
)

__all__ = ["COMPONENT", "router"]
