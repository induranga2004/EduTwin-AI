"""C3: pedagogical relevance of live speech and discussion (keyword baseline)."""

from edusync_c3.relevance import RelevanceLabel, classify, keyword_overlap_score
from edusync_c3.router import router
from edusync_core import Component

COMPONENT = Component(
    name="c3",
    title="Pedagogical relevance",
    description="Classifies speech as relevant or unrelated to learning objectives.",
    router=router,
)

__all__ = ["COMPONENT", "RelevanceLabel", "classify", "keyword_overlap_score", "router"]
