"""Group C: explanation and example support (regex counts, diagram counts)."""

import re

from edusync_c2.difficulty.registry import features_of
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


def explanation_features(
    concept: ConceptRecord, slides: list[SlideRecord], structure_available: bool
) -> list[FeatureValue]:
    """Group C features in registry order."""
    names = features_of(FeatureGroup.EXPLANATION)
    if not slides:
        return [
            _fv(
                n,
                None,
                FeatureStatus.UNAVAILABLE,
                "materials:regex",
                "concept not found in materials",
            )
            for n in names
        ]
    text = "\n".join(f"{s.title or ''}\n{s.text}\n{s.notes}" for s in slides)
    out = [
        _fv(n, float(len(rx.findall(text))), FeatureStatus.COMPUTED, "materials:regex")
        for n, rx in _COUNT_RES.items()
    ]
    if structure_available:
        total = sum((s.picture_count or 0) + (s.chart_count or 0) for s in slides)
        out.append(_fv("expl.diagram_support", float(total), FeatureStatus.COMPUTED, _STRUCT))
    else:
        out.append(
            _fv(
                "expl.diagram_support",
                None,
                FeatureStatus.UNAVAILABLE,
                _STRUCT,
                "PDF: structure not available",
            )
        )
    out.append(
        _fv(
            "expl.semantic_support_score",
            0.0,
            FeatureStatus.MOCKED,
            "mock:todo_semantic_detection",
            "TODO: placeholder 0.0, no semantic detection yet",
        )
    )
    return out
