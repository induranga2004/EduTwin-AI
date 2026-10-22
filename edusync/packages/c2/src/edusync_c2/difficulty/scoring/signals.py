"""Group signals A-D: feature percentiles against the reference, with a direction per feature.

``DIRECTIONS`` says how a feature relates to difficulty: +1 = a higher value makes the concept
harder, -1 = a higher value makes it easier, 0 = shown with its percentile but not scored (the
direction is ambiguous or the feature is a quality measure of the graph, not a difficulty cue).
The score of a feature ("effect") is its percentile for +1 and 1 - percentile for -1, so 1 always
means "harder than every reference concept". A group score is the mean effect of its available
features (status computed, value present, reference table present); None if there is none.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from edusync_c2.difficulty.registry import FEATURE_SPECS
from edusync_c2.difficulty.schemas import FeatureValue
from edusync_c2.difficulty.scoring.reference import Reference

GROUPS = ("A", "B", "C", "D")
GROUP_TITLES = {
    "A": "Language",
    "B": "Prerequisites",
    "C": "Explanation",
    "D": "Density",
}

DIRECTIONS: dict[str, int] = {
    # A language: longer sentences/words, more rare or fragmented vocabulary, more domain terms
    # and a denser concept vocabulary all read as harder.
    "lang.avg_sentence_length": 1,
    "lang.avg_word_length": 1,
    "lang.long_word_ratio": 1,
    "lang.lexical_score": 1,
    "lang.concept_term_density": 1,
    "lang.unfamiliar_vocab_ratio": 1,
    "lang.subword_fragmentation": 1,
    "lang.domain_term_ratio": 1,
    # B prerequisites: more and deeper prerequisites and more dependants (a foundation others
    # build on) mean more to know first. Confidence and curriculum share describe the graph's
    # evidence, not difficulty, so they are shown but not scored.
    "prereq.direct_count": 1,
    "prereq.chain_depth": 1,
    "prereq.transitive_count": 1,
    "prereq.mean_confidence": 0,
    "prereq.curriculum_defined_share": 0,
    "prereq.weighted_depth": 1,
    "prereq.dependents_count": 1,
    # C explanation is INVERTED: more definitions, examples, steps, diagrams and semantic
    # support mean the lecture explains the concept, so it is easier.
    "expl.definition_count": -1,
    "expl.example_count": -1,
    "expl.step_count": -1,
    "expl.diagram_support": -1,
    "expl.semantic_support_score": -1,
    "expl.semantic_definition_count": -1,
    "expl.semantic_example_count": -1,
    # D density: more words, concepts, formulas, code, nesting, shapes, text/table/diagram load
    # and a smaller font are harder. Concept frequency and picture area are ambiguous (a frequent
    # concept is more familiar but also central; pictures can explain or distract): not scored.
    "dens.words_per_slide": 1,
    "dens.concepts_per_slide": 1,
    "dens.formula_presence": 1,
    "dens.code_presence": 1,
    "dens.max_bullet_depth": 1,
    "dens.shape_count_mean": 1,
    "dens.concept_frequency": 0,
    "dens.visual_complexity": 1,
    "dens.text_area_ratio": 1,
    "dens.picture_area_ratio": 0,
    "dens.min_font_pt": -1,
    "dens.diagram_element_count": 1,
    "dens.table_cell_count": 1,
}

FEATURE_GROUP: dict[str, str] = {s.name: s.group.value[0] for s in FEATURE_SPECS}


@dataclass(frozen=True)
class FeatureScore:
    name: str
    group: str
    value: float | None
    status: str
    note: str
    direction: int
    percentile: float | None  # position among the reference concepts, None if not computable
    effect: float | None  # 1 = harder than all reference concepts; None if not scored


def score_features(features: Sequence[FeatureValue], reference: Reference) -> list[FeatureScore]:
    """One ``FeatureScore`` per feature, in the given (registry) order."""
    scores = []
    for f in features:
        direction = DIRECTIONS[f.name]
        status = str(f.status.value)
        percentile = (
            reference.percentile(f.name, f.value)
            if status == "computed" and f.value is not None
            else None
        )
        effect = None
        if percentile is not None and direction != 0:
            effect = percentile if direction > 0 else 1.0 - percentile
        scores.append(
            FeatureScore(
                name=f.name,
                group=FEATURE_GROUP[f.name],
                value=f.value,
                status=status,
                note=f.note,
                direction=direction,
                percentile=percentile,
                effect=effect,
            )
        )
    return scores


def group_scores(scores: Sequence[FeatureScore]) -> dict[str, float | None]:
    """Mean effect of the scored features per group A-D; None when a group has nothing."""
    out: dict[str, float | None] = {}
    for g in GROUPS:
        effects = [s.effect for s in scores if s.group == g and s.effect is not None]
        out[g] = sum(effects) / len(effects) if effects else None
    return out


def top_features(scores: Sequence[FeatureScore], n: int = 3) -> list[FeatureScore]:
    """The ``n`` scored features with the highest effect (ties by name)."""
    scored = [s for s in scores if s.effect is not None]
    return sorted(scored, key=lambda s: (-s.effect, s.name))[:n]
