"""Feature processors (groups A-D) and their registry-ordered combination."""

from edusync_c2.difficulty.features.density import density_features
from edusync_c2.difficulty.features.explanation import explanation_features
from edusync_c2.difficulty.features.language import language_features, lexical_difficulty
from edusync_c2.difficulty.features.prerequisites import prerequisite_features
from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer
from edusync_c2.difficulty.schemas import (
    ConceptRecord,
    FeatureValue,
    InputSource,
    PrerequisiteMetrics,
    SlideRecord,
)

__all__ = [
    "compute_all_features",
    "density_features",
    "explanation_features",
    "language_features",
    "lexical_difficulty",
    "prerequisite_features",
]


def compute_all_features(
    concept: ConceptRecord,
    *,
    all_concepts: list[ConceptRecord],
    slides_by_index: dict[int, SlideRecord],
    structure_available: bool,
    metrics: PrerequisiteMetrics | None,
    graph_has_prerequisite_edges: bool,
    graph_source: InputSource,
    vocab: VocabularyAnalyzer | None = None,
    slide_size: tuple[int, int] | None = None,
) -> list[FeatureValue]:
    """All 26 features in registry order (A, B, C, D).

    `vocab` overrides the shared default vocabulary analyzer (group A); `slide_size` is the deck's
    (width, height) in EMU, `None` when unknown (group D).
    """
    indexes = sorted(set(concept.slide_indexes))
    slides = [slides_by_index[i] for i in indexes if i in slides_by_index]
    return [
        *language_features(concept, slides, vocab),
        *prerequisite_features(concept, metrics, graph_has_prerequisite_edges, graph_source),
        *explanation_features(concept, slides, structure_available),
        *density_features(
            concept, slides, all_concepts, structure_available, graph_source, slide_size
        ),
    ]
