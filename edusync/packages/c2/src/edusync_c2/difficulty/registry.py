"""The ordered feature registry: the single source of truth for feature names and order.

Every feature processor must return exactly the names of its group below, always, even when a
value cannot be produced (then status ``unavailable`` or ``mocked``). Fusion uses this order.
maturity: implemented = built; mock = placeholder value; planned = not built yet (always
unavailable). requires: text = computed from the lecture text; pptx_structure = PPTX only (PDF
unavailable); c1_graph = needs the C1 graph (mocked graph -> status mocked). The legacy ``phase1``
field is computed from both (see ``FeatureSpec``).
"""

import hashlib

from edusync_c2.difficulty.schemas import FeatureGroup as G
from edusync_c2.difficulty.schemas import FeatureSpec, Maturity, Requires

# Feature schema version. Bump it whenever a feature name, its group or the order changes.
#   0.1  Phase 1: 24 features (the first registry)
#   0.2  Phase 2 M1: vocabulary features (26 features) and the versioned schema hash
#   0.3  pre-registers the nine Phase 2 features (35 features), all planned
# To be frozen as 1.0 at the end of Phase 2. A trained model records version and hash.
FEATURE_SCHEMA_VERSION = "0.3"


def _f(
    name: str, group: G, maturity: Maturity, requires: Requires, description: str
) -> FeatureSpec:
    return FeatureSpec(
        name=name, group=group, maturity=maturity, requires=requires, description=description
    )


FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # A: technical vocabulary and language complexity
    _f(
        "lang.avg_sentence_length",
        G.LANGUAGE,
        "implemented",
        "text",
        "Mean words per sentence on the slides.",
    ),
    _f(
        "lang.avg_word_length",
        G.LANGUAGE,
        "implemented",
        "text",
        "Mean letters per word on the slides.",
    ),
    _f(
        "lang.long_word_ratio",
        G.LANGUAGE,
        "implemented",
        "text",
        "Share of words with 9+ letters (technical proxy).",
    ),
    _f(
        "lang.lexical_score",
        G.LANGUAGE,
        "implemented",
        "text",
        "Transparent lexical difficulty heuristic in [0, 1].",
    ),
    _f(
        "lang.concept_term_density",
        G.LANGUAGE,
        "implemented",
        "text",
        "Concept mentions per 100 words on its slides.",
    ),
    _f(
        "lang.unfamiliar_vocab_ratio",
        G.LANGUAGE,
        "implemented",
        "text",
        "Share of words that need 2+ MiniLM word pieces.",
    ),
    _f(
        "lang.subword_fragmentation",
        G.LANGUAGE,
        "implemented",
        "text",
        "Mean MiniLM word pieces per word.",
    ),
    _f(
        "lang.domain_term_ratio",
        G.LANGUAGE,
        "implemented",
        "text",
        "Share of words in the curated CS/IT lexicon.",
    ),
    # B: prerequisite and concept complexity (from the C1 graph, prerequisite edges only)
    _f(
        "prereq.direct_count",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "Number of direct prerequisites.",
    ),
    _f(
        "prereq.chain_depth",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "Longest prerequisite chain below the concept.",
    ),
    _f(
        "prereq.transitive_count",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "All distinct (transitive) prerequisites.",
    ),
    _f(
        "prereq.mean_confidence",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "Mean confidence of direct prerequisites.",
    ),
    _f(
        "prereq.curriculum_defined_share",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "Share of direct prerequisites from the curriculum.",
    ),
    _f(
        "prereq.weighted_depth",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "Sum of confidences along the heaviest prerequisite chain.",
    ),
    _f(
        "prereq.dependents_count",
        G.PREREQUISITES,
        "implemented",
        "c1_graph",
        "Number of concepts that directly depend on this one.",
    ),
    # C: explanation and example support
    _f(
        "expl.definition_count",
        G.EXPLANATION,
        "implemented",
        "text",
        "Definition phrases on the concept's slides.",
    ),
    _f(
        "expl.example_count",
        G.EXPLANATION,
        "implemented",
        "text",
        "Worked-example phrases on the concept's slides.",
    ),
    _f(
        "expl.step_count",
        G.EXPLANATION,
        "implemented",
        "text",
        "Step-by-step markers (Step 1, numbered lists).",
    ),
    _f(
        "expl.diagram_support",
        G.EXPLANATION,
        "implemented",
        "pptx_structure",
        "Pictures + charts on its slides (PPTX).",
    ),
    _f(
        "expl.semantic_support_score",
        G.EXPLANATION,
        "mock",
        "text",
        "TODO: semantic definition/example detection.",
    ),
    _f(
        "expl.semantic_definition_count",
        G.EXPLANATION,
        "planned",
        "text",
        "Semantic (embedding-based) definition detections on the concept's slides.",
    ),
    _f(
        "expl.semantic_example_count",
        G.EXPLANATION,
        "planned",
        "text",
        "Semantic worked-example detections on the concept's slides.",
    ),
    # D: information density and structure
    _f("dens.words_per_slide", G.DENSITY, "implemented", "text", "Mean words per concept slide."),
    _f(
        "dens.concepts_per_slide",
        G.DENSITY,
        "implemented",
        "text",
        "Mean distinct C1 concepts per concept slide.",
    ),
    _f(
        "dens.formula_presence",
        G.DENSITY,
        "implemented",
        "text",
        "Share of concept slides that look like formulas.",
    ),
    _f(
        "dens.code_presence",
        G.DENSITY,
        "implemented",
        "text",
        "Share of concept slides that look like code.",
    ),
    _f(
        "dens.max_bullet_depth",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Deepest bullet level on its slides (PPTX).",
    ),
    _f(
        "dens.shape_count_mean",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Mean shapes per concept slide (PPTX).",
    ),
    _f(
        "dens.concept_frequency",
        G.DENSITY,
        "implemented",
        "c1_graph",
        "C1 node frequency of the concept.",
    ),
    _f(
        "dens.visual_complexity",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "0-1 heuristic: .35 sat(media,4) + .25 sat(diagram,15) + .25 area + .15 sat(info,30).",
    ),
    _f(
        "dens.text_area_ratio",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Mean share of slide area covered by text shapes (union, clamped 0-1).",
    ),
    _f(
        "dens.picture_area_ratio",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Mean share of slide area covered by pictures (union, clamped 0-1).",
    ),
    _f(
        "dens.min_font_pt",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Smallest explicit font size on the concept's slides, in points.",
    ),
    _f(
        "dens.diagram_element_count",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Mean diagram elements (autoshapes, connectors, freeforms, SmartArt) per concept slide.",
    ),
    _f(
        "dens.table_cell_count",
        G.DENSITY,
        "implemented",
        "pptx_structure",
        "Mean table cells (rows x columns) per concept slide.",
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(s.name for s in FEATURE_SPECS)

PHASE1_TODO: tuple[str, ...] = (
    "Real C1 connection (C1ApiGraphProvider); the graph is a mock unless supplied by the caller.",
    "Concept filter: alias merging (TDE = Transparent Data Encryption) and generic phrases.",
    "Feature C: semantic definition/example detection (expl.semantic_support_score is a mock; "
    "expl.semantic_definition_count and expl.semantic_example_count are planned).",
    "MiniLMFusionModel and learned feature fusion (Phase 3, trained on lecturer labels).",
    "Calibrated confidence and probabilities (never invented before a trained, evaluated model).",
    "Prediction persistence for Phase 5 comparison with live comprehension.",
)


def features_of(group: G) -> tuple[str, ...]:
    return tuple(s.name for s in FEATURE_SPECS if s.group is group)


def spec_of(name: str) -> FeatureSpec:
    """The registry spec of ``name``; ``KeyError`` when the name is not registered."""
    for spec in FEATURE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(name)


def feature_schema_hash(specs: tuple[FeatureSpec, ...] = FEATURE_SPECS) -> str:
    """sha256 of the schema version and the ordered feature names and groups.

    Payload: ``edusync-c2-feature-schema\\nversion=<v>\\n`` then one ``name|group`` line per
    feature. ``maturity``, ``requires`` and ``description`` are metadata and are not hashed.
    A trained model records this hash in its artifacts so a registry change that would silently
    shift feature positions is detected (see ``OnnxFusionModel.from_artifacts``).
    """
    lines = "\n".join(f"{s.name}|{s.group.value}" for s in specs)
    payload = f"edusync-c2-feature-schema\nversion={FEATURE_SCHEMA_VERSION}\n{lines}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
