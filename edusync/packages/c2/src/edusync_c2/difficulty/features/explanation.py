"""Group C: explanation and example support (regex counts, diagram counts, semantic counts).

The three ``expl.semantic_*`` features come only from the semantic scorer (embedding similarity to
prototype sentences, see ``features/semantic.py``). Without a scorer (switched off or its model
could not be loaded) they are unavailable: regex values are never substituted and no placeholder
is invented. The thresholds behind the counts are heuristic and not validated.
"""

import re

from edusync_c2.difficulty.features._common import assemble, built_features_of
from edusync_c2.difficulty.features.semantic import SemanticScorer, split_sentences
from edusync_c2.difficulty.schemas import (
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    FeatureValue,
    SlideRecord,
)

# Definition cues; "is a/an <word>" is deliberately simple.
DEFINITION_RE = re.compile(
    r"\b(?:is\s+defined\s+as|refers?\s+to|is\s+called|is\s+known\s+as|means|denotes"
    r"|is\s+an?\s+[a-z]+)\b",
    re.IGNORECASE,
)
# Example cues.
EXAMPLE_RE = re.compile(
    r"\b(?:for\s+example|for\s+instance|worked\s+example|such\s+as)\b|\be\.g\.|\bexample\s*:",
    re.IGNORECASE,
)
# "Step 1", "step 2:", numbered list lines ("1." / "2)"), and "first, ... then" within a line.
STEP_RE = re.compile(
    r"\bstep\s+\d+\b|^\s*\d+[.)]\s+\S|\bfirst\b[^.\n]*?,[^.\n]*?\bthen\b",
    re.IGNORECASE | re.MULTILINE,
)
_COUNT_RES = {
    "expl.definition_count": DEFINITION_RE,
    "expl.example_count": EXAMPLE_RE,
    "expl.step_count": STEP_RE,
}
_STRUCT = "materials:pptx_structure"
SEMANTIC_NAMES = (
    "expl.semantic_definition_count",
    "expl.semantic_example_count",
    "expl.semantic_support_score",
)
_SEMANTIC_SOURCE = "semantic:minilm"


def _fv(
    name: str, value: float | None, status: FeatureStatus, source: str, note: str = ""
) -> FeatureValue:
    return FeatureValue(
        name=name,
        group=FeatureGroup.EXPLANATION,
        value=value,
        status=status,
        source=source,
        note=note,
    )


def _slide_text(slides: list[SlideRecord]) -> str:
    """Slide text plus notes (SlideRecord.text already holds the title, so it is not added)."""
    return "\n".join(f"{s.text}\n{s.notes}" for s in slides)


def _semantic_features(
    slides: list[SlideRecord], semantic: SemanticScorer | None, semantic_status: str
) -> dict[str, FeatureValue]:
    """The three semantic features; all unavailable (with the reason) when there is no scorer."""
    if semantic is None:
        off = semantic_status == "off"
        source = "semantic:off" if off else _SEMANTIC_SOURCE
        note = "semantic scorer is off" if off else "semantic scorer unavailable"
        return {n: _fv(n, None, FeatureStatus.UNAVAILABLE, source, note) for n in SEMANTIC_NAMES}
    report = semantic.score_report(split_sentences(_slide_text(slides)))
    definitions, examples = semantic.count_semantic(report.scores)
    note = "sentence budget reached: later sentences were not scored" if report.truncated else ""
    support = (min(definitions, 1) + min(examples, 1)) / 2
    values = {
        "expl.semantic_definition_count": float(definitions),
        "expl.semantic_example_count": float(examples),
        "expl.semantic_support_score": support,
    }
    return {n: _fv(n, v, FeatureStatus.COMPUTED, _SEMANTIC_SOURCE, note) for n, v in values.items()}


def explanation_features(
    concept: ConceptRecord,
    slides: list[SlideRecord],
    structure_available: bool,
    semantic: SemanticScorer | None = None,
    semantic_status: str = "off",
) -> list[FeatureValue]:
    """Group C features in registry order.

    `semantic` is the per-request scorer (None = off or unavailable, see `semantic_status`:
    "off" or an "minilm:..." status string).
    """
    if not slides:
        return assemble(
            FeatureGroup.EXPLANATION,
            {
                n: _fv(
                    n,
                    None,
                    FeatureStatus.UNAVAILABLE,
                    "materials:regex",
                    "concept not found in materials",
                )
                for n in built_features_of(FeatureGroup.EXPLANATION)
            },
        )
    text = _slide_text(slides)
    out = {
        n: _fv(n, float(len(rx.findall(text))), FeatureStatus.COMPUTED, "materials:regex")
        for n, rx in _COUNT_RES.items()
    }
    if structure_available:
        total = sum((s.picture_count or 0) + (s.chart_count or 0) for s in slides)
        out["expl.diagram_support"] = _fv(
            "expl.diagram_support", float(total), FeatureStatus.COMPUTED, _STRUCT
        )
    else:
        out["expl.diagram_support"] = _fv(
            "expl.diagram_support",
            None,
            FeatureStatus.UNAVAILABLE,
            _STRUCT,
            "PDF: structure not available",
        )
    out.update(_semantic_features(slides, semantic, semantic_status))
    return assemble(FeatureGroup.EXPLANATION, out)
