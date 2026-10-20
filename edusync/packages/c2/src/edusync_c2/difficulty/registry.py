"""The ordered feature registry: the single source of truth for feature names and order.

Every feature processor must return exactly the names of its group below, always, even when a
value cannot be produced (then status ``unavailable`` or ``mocked``). Fusion uses this order.
phase1: real = computed from the lecture file; real_needs_structure = PPTX only (PDF unavailable);
graph = needs the C1 graph (mocked graph -> status mocked); mock = placeholder; todo = not built.
"""

from typing import Literal

from edusync_c2.difficulty.schemas import FeatureGroup as G
from edusync_c2.difficulty.schemas import FeatureSpec

Phase1 = Literal["real", "real_needs_structure", "graph", "mock", "todo"]


def _f(name: str, group: G, phase1: Phase1, description: str) -> FeatureSpec:
    return FeatureSpec(name=name, group=group, phase1=phase1, description=description)


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # A: technical vocabulary and language complexity
    _f("lang.avg_sentence_length", G.LANGUAGE, "real", "Mean words per sentence on the slides."),
    _f("lang.avg_word_length", G.LANGUAGE, "real", "Mean letters per word on the slides."),
    _f(
        "lang.long_word_ratio",
        G.LANGUAGE,
        "real",
        "Share of words with 9+ letters (technical proxy).",
    ),
    _f(
        "lang.lexical_score",
        G.LANGUAGE,
        "real",
        "Transparent lexical difficulty heuristic in [0, 1].",
    ),
    _f(
        "lang.concept_term_density",
        G.LANGUAGE,
        "real",
        "Concept mentions per 100 words on its slides.",
    ),
    _f(
        "lang.unfamiliar_vocab_ratio",
        G.LANGUAGE,
        "mock",
        "TODO: share of words outside a domain list.",
    ),
    # B: prerequisite and concept complexity (from the C1 graph, prerequisite edges only)
    _f("prereq.direct_count", G.PREREQUISITES, "graph", "Number of direct prerequisites."),
    _f(
        "prereq.chain_depth",
        G.PREREQUISITES,
        "graph",
        "Longest prerequisite chain below the concept.",
    ),
    _f(
        "prereq.transitive_count",
        G.PREREQUISITES,
        "graph",
        "All distinct (transitive) prerequisites.",
    ),
    _f(
        "prereq.mean_confidence",
        G.PREREQUISITES,
        "graph",
        "Mean confidence of direct prerequisites.",
    ),
    _f(
        "prereq.curriculum_defined_share",
        G.PREREQUISITES,
        "graph",
        "Share of direct prerequisites from the curriculum.",
    ),
    # C: explanation and example support
    _f(
        "expl.definition_count",
        G.EXPLANATION,
        "real",
        "Definition phrases on the concept's slides.",
    ),
    _f(
        "expl.example_count",
        G.EXPLANATION,
        "real",
        "Worked-example phrases on the concept's slides.",
    ),
    _f("expl.step_count", G.EXPLANATION, "real", "Step-by-step markers (Step 1, numbered lists)."),
    _f(
        "expl.diagram_support",
        G.EXPLANATION,
        "real_needs_structure",
        "Pictures + charts on its slides (PPTX).",
    ),
    _f(
        "expl.semantic_support_score",
        G.EXPLANATION,
        "mock",
        "TODO: semantic definition/example detection.",
    ),
    # D: information density and structure
    _f("dens.words_per_slide", G.DENSITY, "real", "Mean words per concept slide."),
    _f(
        "dens.concepts_per_slide", G.DENSITY, "real", "Mean distinct C1 concepts per concept slide."
    ),
    _f(
        "dens.formula_presence",
        G.DENSITY,
        "real",
        "Share of concept slides that look like formulas.",
    ),
    _f("dens.code_presence", G.DENSITY, "real", "Share of concept slides that look like code."),
    _f(
        "dens.max_bullet_depth",
        G.DENSITY,
        "real_needs_structure",
        "Deepest bullet level on its slides (PPTX).",
    ),
    _f(
        "dens.shape_count_mean",
        G.DENSITY,
        "real_needs_structure",
        "Mean shapes per concept slide (PPTX).",
    ),
    _f("dens.concept_frequency", G.DENSITY, "graph", "C1 node frequency of the concept."),
    _f("dens.visual_complexity", G.DENSITY, "todo", "TODO: diagram/visual complexity analysis."),
)

FEATURE_NAMES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)

PHASE1_TODO: tuple[str, ...] = (
    "Real C1 connection (C1ApiGraphProvider); the graph is a mock unless supplied by the caller.",
    "Concept filter: alias merging (TDE = Transparent Data Encryption) and generic phrases.",
    "Feature A: unfamiliar-vocabulary score from a domain word list.",
    "Feature C: semantic definition/example detection.",
    "Feature D: visual/diagram complexity analysis.",
    "MiniLMFusionModel and learned feature fusion (Phase 3, trained on lecturer labels).",
    "Calibrated confidence and probabilities (never invented before a trained, evaluated model).",
    "Prediction persistence for Phase 5 comparison with live comprehension.",
)


def features_of(group: G) -> tuple[str, ...]:
    return tuple(s.name for s in FEATURE_SPECS if s.group is group)
