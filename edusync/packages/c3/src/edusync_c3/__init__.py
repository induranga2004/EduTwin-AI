"""C3: pedagogical relevance of live speech and discussion (keyword baseline)."""

from edusync_c3.keep_filter import KeepFilter, ModelUnavailableError
from edusync_c3.relevance import RelevanceLabel, classify, keyword_overlap_score
from edusync_c3.router import router
from edusync_c3.sentences import split_sentences
from edusync_core import Component

COMPONENT = Component(
    name="c3",
    title="Pedagogical relevance",
    description=(
        "Classifies speech as relevant or unrelated to learning objectives and filters "
        "lecture transcripts down to teaching content."
    ),
    router=router,
)

__all__ = [
    "COMPONENT",
    "KeepFilter",
    "ModelUnavailableError",
    "RelevanceLabel",
    "classify",
    "keyword_overlap_score",
    "router",
    "split_sentences",
]
