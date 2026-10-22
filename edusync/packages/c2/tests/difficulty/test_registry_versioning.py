"""Feature schema version, hash payload, maturity/requires/phase1 and baseline constants."""

import hashlib
import itertools

import pytest

from edusync_c2.difficulty import models
from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SPECS,
    feature_schema_hash,
    spec_of,
)
from edusync_c2.difficulty.schemas import FeatureGroup, FeatureSpec

# The mapping the registry used before maturity/requires existed (phase1 per feature name).
LEGACY_PHASE1 = {
    **dict.fromkeys(
        (
            "lang.avg_sentence_length",
            "lang.avg_word_length",
            "lang.long_word_ratio",
            "lang.lexical_score",
            "lang.concept_term_density",
            "lang.unfamiliar_vocab_ratio",
            "lang.subword_fragmentation",
            "lang.domain_term_ratio",
            "expl.definition_count",
            "expl.example_count",
            "expl.step_count",
            "dens.words_per_slide",
            "dens.concepts_per_slide",
            "dens.formula_presence",
            "dens.code_presence",
        ),
        "real",
    ),
    **dict.fromkeys(
        ("expl.diagram_support", "dens.max_bullet_depth", "dens.shape_count_mean"),
        "real_needs_structure",
    ),
    **dict.fromkeys(
        (
            "prereq.direct_count",
            "prereq.chain_depth",
            "prereq.transitive_count",
            "prereq.mean_confidence",
            "prereq.curriculum_defined_share",
            "dens.concept_frequency",
        ),
        "graph",
    ),
    "expl.semantic_support_score": "mock",
    "dens.visual_complexity": "todo",
    # pre-registered in schema 0.3, all planned
    **dict.fromkeys(
        (
            "prereq.weighted_depth",
            "prereq.dependents_count",
            "expl.semantic_definition_count",
            "expl.semantic_example_count",
            "dens.text_area_ratio",
            "dens.picture_area_ratio",
            "dens.min_font_pt",
            "dens.diagram_element_count",
            "dens.table_cell_count",
        ),
        "todo",
    ),
}


def _spec(maturity, requires, description="d"):
    return FeatureSpec(
        name="x.y",
        group=FeatureGroup.DENSITY,
        maturity=maturity,
        requires=requires,
        description=description,
    )


def test_version_constant():
    assert FEATURE_SCHEMA_VERSION == "0.3"


@pytest.mark.parametrize(
    ("maturity", "requires", "expected"),
    [
        ("implemented", "text", "real"),
        ("implemented", "pptx_structure", "real_needs_structure"),
        ("implemented", "c1_graph", "graph"),
        *[("mock", r, "mock") for r in ("text", "pptx_structure", "c1_graph")],
        *[("planned", r, "todo") for r in ("text", "pptx_structure", "c1_graph")],
    ],
)
def test_phase1_is_computed(maturity, requires, expected):
    spec = _spec(maturity, requires)
    assert spec.phase1 == expected
    assert spec.model_dump()["phase1"] == expected
    assert spec.model_dump(mode="json")["maturity"] == maturity


def test_legacy_phase1_parity():
    assert {s.name: s.phase1 for s in FEATURE_SPECS} == LEGACY_PHASE1
    assert set(LEGACY_PHASE1) == set(FEATURE_NAMES)


def test_spec_of_and_unknown():
    assert spec_of("dens.visual_complexity").maturity == "planned"
    with pytest.raises(KeyError):
        spec_of("nope")


def test_hash_payload_has_versioned_header():
    lines = "\n".join(f"{s.name}|{s.group.value}" for s in FEATURE_SPECS)
    payload = f"edusync-c2-feature-schema\nversion={FEATURE_SCHEMA_VERSION}\n{lines}"
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert feature_schema_hash() == expected
    assert feature_schema_hash(FEATURE_SPECS) == expected


def test_hash_ignores_metadata_but_not_names_or_order():
    base = feature_schema_hash()
    changed = tuple(
        s.model_copy(update={"description": "other", "maturity": "mock", "requires": "text"})
        for s in FEATURE_SPECS
    )
    assert feature_schema_hash(changed) == base
    swapped = (FEATURE_SPECS[1], FEATURE_SPECS[0], *FEATURE_SPECS[2:])
    assert feature_schema_hash(swapped) != base
    renamed = (FEATURE_SPECS[0].model_copy(update={"name": "lang.other"}), *FEATURE_SPECS[1:])
    assert feature_schema_hash(renamed) != base


def test_all_combinations_have_a_phase1():
    for m, r in itertools.product(
        ("implemented", "mock", "planned"), ("text", "pptx_structure", "c1_graph")
    ):
        assert _spec(m, r).phase1


def test_current_constants_are_valid():
    models._validate_constants(
        models.RAISING_WEIGHTS, models._NORMALISATION, models.SUPPORT_FEATURES, FEATURE_NAMES
    )


def test_validate_constants_key_mismatch():
    with pytest.raises(ValueError, match="keys differ: b"):
        models._validate_constants({"a": 1.0}, {"a": (0.0, 1.0), "b": (0.0, 1.0)}, (), ("a", "b"))


def test_validate_constants_unknown_names():
    with pytest.raises(ValueError, match="not in the registry: ghost, spooky"):
        models._validate_constants(
            {"a": 1.0, "ghost": 1.0}, {"a": (0.0, 1.0), "ghost": (0.0, 1.0)}, ("spooky",), ("a",)
        )


@pytest.mark.parametrize("span", [0.0, -1.0])
def test_validate_constants_bad_span(span):
    with pytest.raises(ValueError, match="span must be > 0 for: a"):
        models._validate_constants({"a": 1.0}, {"a": (0.0, span)}, (), ("a",))
