"""Group D: information density and slide structure."""

import re
from collections.abc import Callable
from fractions import Fraction

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
_NO_SIZE = "slide size unknown"
_NO_GEOMETRY = "geometry unknown"
_NO_FONT = "no explicit font size"
_NOT_FOUND = "concept not found in materials"

# dens.visual_complexity = sum(weight * component) / sum(weight of the known components).
# Heuristic and unvalidated: the weights and saturation points are a first guess, not fitted.
_VC_WEIGHTS: dict[str, Fraction] = {
    "media": Fraction(35, 100),
    "diagram": Fraction(25, 100),
    "visual_area": Fraction(25, 100),
    "info": Fraction(15, 100),
}
# sat(x, k) = min(x / k, 1): the per-slide mean at which a component counts as "full".
# visual_area is already a ratio in [0, 1] and has no saturation point.
_VC_SATURATION: dict[str, int] = {"media": 4, "diagram": 15, "info": 30}
_STRUCTURE_NAMES = (
    "dens.text_area_ratio",
    "dens.picture_area_ratio",
    "dens.min_font_pt",
    "dens.diagram_element_count",
    "dens.table_cell_count",
    "dens.visual_complexity",
)
_ZERO = Fraction(0)
_ONE = Fraction(1)


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
    slide_size: tuple[int, int] | None = None,
) -> list[FeatureValue]:
    """Group D features in registry order.

    ``slide_size`` is the deck's (width, height) in EMU, ``None`` when unknown (PDF).
    """
    mocked = graph_source.kind == "mock"
    freq = _fv(
        "dens.concept_frequency",
        float(concept.frequency),
        FeatureStatus.MOCKED if mocked else _C,
        "mock:c1_graph" if mocked else "c1_graph:node_frequency",
        "frequency comes from a mock graph" if mocked else "",
    )
    if not slides:
        note = _NOT_FOUND
        return _assemble(
            [
                _fv("dens.words_per_slide", None, _U, _SRC, note),
                _fv("dens.concepts_per_slide", None, _U, _SRC, note),
                _fv("dens.formula_presence", None, _U, _SRC, note),
                _fv("dens.code_presence", None, _U, _SRC, note),
                _fv("dens.max_bullet_depth", None, _U, _STRUCT, note),
                _fv("dens.shape_count_mean", None, _U, _STRUCT, note),
                *(_fv(name, None, _U, _STRUCT, note) for name in _STRUCTURE_NAMES),
                freq,
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
        out.extend(_geometry_features(slides, _area(slide_size)))
    else:
        out.append(_fv("dens.max_bullet_depth", None, _U, _STRUCT, _PDF_NOTE))
        out.append(_fv("dens.shape_count_mean", None, _U, _STRUCT, _PDF_NOTE))
        out.extend(_fv(name, None, _U, _STRUCT, _PDF_NOTE) for name in _STRUCTURE_NAMES)
    return _assemble([*out, freq])


def _area(slide_size: tuple[int, int] | None) -> int | None:
    """The slide area in EMU squared; ``None`` when the size is unknown or not positive."""
    if slide_size is None or slide_size[0] <= 0 or slide_size[1] <= 0:
        return None
    return slide_size[0] * slide_size[1]


def _unreadable(slides: list[SlideRecord], bound: str) -> str:
    """Note for values built from slides with shapes whose geometry could not be read."""
    count = sum(s.geometry_unknown_count or 0 for s in slides)
    return f"{bound}: {count} shape(s) with unreadable geometry" if count else ""


def _ratio(emu: int, area: int) -> Fraction:
    return min(max(Fraction(emu, area), _ZERO), _ONE)


def _mean(values: list[Fraction]) -> Fraction:
    return sum(values, _ZERO) / len(values)


def _geometry_features(slides: list[SlideRecord], area: int | None) -> list[FeatureValue]:
    return [
        _ratio_feature("dens.text_area_ratio", slides, "text_area_emu", area),
        _ratio_feature("dens.picture_area_ratio", slides, "picture_area_emu", area),
        _min_font_feature(slides),
        _count_feature("dens.diagram_element_count", slides, "diagram_element_count"),
        _count_feature("dens.table_cell_count", slides, "table_cell_count"),
        _visual_complexity(slides, area),
    ]


def _ratio_feature(
    name: str, slides: list[SlideRecord], attr: str, area: int | None
) -> FeatureValue:
    if area is None:
        return _fv(name, None, _U, _STRUCT, _NO_SIZE)
    used = [s for s in slides if getattr(s, attr) is not None]
    if not used:
        return _fv(name, None, _U, _STRUCT, _NO_GEOMETRY)
    mean = _mean([_ratio(getattr(s, attr), area) for s in used])
    return _fv(name, float(mean), _C, _STRUCT, _unreadable(used, "lower bound"))


def _min_font_feature(slides: list[SlideRecord]) -> FeatureValue:
    name = "dens.min_font_pt"
    used = [s for s in slides if s.min_font_pt is not None]
    if not used:
        return _fv(name, None, _U, _STRUCT, _NO_FONT)
    # Shapes with unreadable fonts could hold a smaller size, so the minimum is an upper bound.
    smallest = min(s.min_font_pt for s in used)
    return _fv(name, float(smallest), _C, _STRUCT, _unreadable(used, "upper bound"))


def _count_feature(name: str, slides: list[SlideRecord], attr: str) -> FeatureValue:
    used = [s for s in slides if getattr(s, attr) is not None]
    if not used:
        return _fv(name, None, _U, _STRUCT, _NO_GEOMETRY)
    mean = _mean([Fraction(getattr(s, attr)) for s in used])
    return _fv(name, float(mean), _C, _STRUCT, _unreadable(used, "lower bound"))


def _total(*parts: int | None) -> Fraction | None:
    """The sum of the parts, ``None`` when any part is unknown."""
    known = [p for p in parts if p is not None]
    return Fraction(sum(known)) if len(known) == len(parts) else None


def _component(
    slides: list[SlideRecord], value: Callable[[SlideRecord], Fraction | None]
) -> tuple[Fraction, list[SlideRecord]] | None:
    """Mean of ``value`` over the slides where it is known, and those slides."""
    pairs = [(s, v) for s in slides if (v := value(s)) is not None]
    if not pairs:
        return None
    return _mean([v for _, v in pairs]), [s for s, _ in pairs]


def _visual_complexity(slides: list[SlideRecord], area: int | None) -> FeatureValue:
    name = "dens.visual_complexity"

    def visual_area(s: SlideRecord) -> Fraction | None:
        if area is None or s.visual_area_emu is None:
            return None
        return _ratio(s.visual_area_emu, area)

    parts = {
        "media": _component(
            slides, lambda s: _total(s.picture_count, s.chart_count, s.smartart_count)
        ),
        "diagram": _component(slides, lambda s: _total(s.diagram_element_count)),
        "visual_area": _component(slides, visual_area),
        "info": _component(slides, lambda s: _total(s.table_cell_count, s.shape_count)),
    }
    known = {k: p for k, p in parts.items() if p is not None}
    if not known:
        return _fv(name, None, _U, _STRUCT, _NO_SIZE if area is None else _NO_GEOMETRY)
    total = _ZERO
    weight = _ZERO
    for key, (mean, _) in known.items():
        sat = _VC_SATURATION.get(key)
        level = mean if sat is None else min(mean / sat, _ONE)
        total += _VC_WEIGHTS[key] * level
        weight += _VC_WEIGHTS[key]
    used = {s.index: s for _, group in known.values() for s in group}
    notes = [_unreadable(list(used.values()), "lower bound")]
    missing = [k for k in _VC_WEIGHTS if k not in known]
    if missing:
        notes.append(f"unknown components: {', '.join(missing)}")
    return _fv(name, float(total / weight), _C, _STRUCT, "; ".join(x for x in notes if x))


def _assemble(values: list[FeatureValue]) -> list[FeatureValue]:
    return assemble(FeatureGroup.DENSITY, {v.name: v for v in values})
