"""Group D: information density and slide structure."""

import re

from edusync_c2.difficulty.features._common import assemble
from edusync_c2.difficulty.schemas import (
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    FeatureValue,
    InputSource,
    SlideRecord,
)

_WORD_RE = re.compile(r"\w+")
_PDF_NOTE = "PDF: structure not available"
_SRC = "materials:regex"
_STRUCT = "materials:pptx_structure"
_C = FeatureStatus.COMPUTED
_U = FeatureStatus.UNAVAILABLE


def _fv(
    name: str, value: float | None, status: FeatureStatus, source: str, note: str = ""
) -> FeatureValue:
    return FeatureValue(
        name=name,
        group=FeatureGroup.DENSITY,
        value=value,
        status=status,
        source=source,
        note=note,
    )


def density_features(
    concept: ConceptRecord,
    slides: list[SlideRecord],
    all_concepts: list[ConceptRecord],
    structure_available: bool,
    graph_source: InputSource,
) -> list[FeatureValue]:
    """Group D features in registry order."""
    mocked = graph_source.kind == "mock"
    freq = _fv(
        "dens.concept_frequency",
        float(concept.frequency),
        FeatureStatus.MOCKED if mocked else _C,
        "mock:c1_graph" if mocked else "c1_graph:node_frequency",
        "frequency comes from a mock graph" if mocked else "",
    )
    visual = _fv(
        "dens.visual_complexity",
        None,
        _U,
        "todo:visual_complexity",
        "TODO: visual complexity analysis not implemented",
    )
    if not slides:
        note = "concept not found in materials"
        return _assemble(
            [
                _fv("dens.words_per_slide", None, _U, _SRC, note),
                _fv("dens.concepts_per_slide", None, _U, _SRC, note),
                _fv("dens.formula_presence", None, _U, _SRC, note),
                _fv("dens.code_presence", None, _U, _SRC, note),
                _fv("dens.max_bullet_depth", None, _U, _STRUCT, note),
                _fv("dens.shape_count_mean", None, _U, _STRUCT, note),
                freq,
                visual,
            ]
        )
    n = len(slides)
    words = sum(len(_WORD_RE.findall(s.text)) for s in slides) / n
    per_slide = sum(sum(1 for k in all_concepts if s.index in k.slide_indexes) for s in slides) / n
    out = [
        _fv("dens.words_per_slide", words, _C, _SRC),
        _fv("dens.concepts_per_slide", per_slide, _C, _SRC),
        _fv("dens.formula_presence", sum(s.formula_hint for s in slides) / n, _C, _SRC),
        _fv("dens.code_presence", sum(s.code_hint for s in slides) / n, _C, _SRC),
    ]
    if structure_available:
        levels = [p.level for s in slides for p in s.paragraphs if p.level is not None]
        out.append(_fv("dens.max_bullet_depth", float(max(levels, default=0)), _C, _STRUCT))
        counts = [s.shape_count for s in slides]
        if any(x is None for x in counts):
            out.append(_fv("dens.shape_count_mean", None, _U, _STRUCT, "shape count unknown"))
        else:
            out.append(_fv("dens.shape_count_mean", sum(counts) / n, _C, _STRUCT))
    else:
        out.append(_fv("dens.max_bullet_depth", None, _U, _STRUCT, _PDF_NOTE))
        out.append(_fv("dens.shape_count_mean", None, _U, _STRUCT, _PDF_NOTE))
    return _assemble([*out, freq, visual])


def _assemble(values: list[FeatureValue]) -> list[FeatureValue]:
    return assemble(FeatureGroup.DENSITY, {v.name: v for v in values})
