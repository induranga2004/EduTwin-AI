"""Concept list: C1 nodes aligned to slides, then passed through a (mock) concept filter."""

import re
from typing import Protocol

from edusync_c2.difficulty.ids import casefold_id
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    ConceptRecord,
    GraphContext,
    MaterialsDoc,
    SlideRecord,
)


def normalize(text: str) -> str:
    """Casefold, collapse whitespace and strip (the same rule C1 uses for concept ids)."""
    return casefold_id(text)


def _phrase_pattern(term: str) -> re.Pattern[str] | None:
    words = term.split()
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def locate_slides(label: str, concept_id: str, slides: list[SlideRecord]) -> list[int]:
    """Indexes of slides whose title or text contains the label or id as a whole phrase."""
    patterns = [p for p in (_phrase_pattern(label), _phrase_pattern(concept_id)) if p]
    found = {
        s.index for s in slides if any(p.search(f"{s.title or ''}\n{s.text}") for p in patterns)
    }
    return sorted(found)


class ConceptFilter(Protocol):
    name: str

    def apply(self, concepts: list[ConceptRecord]) -> list[ConceptRecord]: ...


class MockConceptFilter:
    """Keeps every concept. TODO: replace with a real filter."""

    name = "mock_keep_all"
    NOTE = (
        "mock: no filtering (TODO: alias merging such as TDE = Transparent Data Encryption, "
        "generic phrases such as Best Practices)"
    )

    def apply(self, concepts: list[ConceptRecord]) -> list[ConceptRecord]:
        update = {"filter_status": "kept", "filter_note": self.NOTE}
        return [c.model_copy(update=update) for c in concepts]


def build_concepts(
    ctx: GraphContext, materials: MaterialsDoc, concept_filter: ConceptFilter | None = None
) -> list[ConceptRecord]:
    """Concepts from C1 nodes (fallback: C1 concept names), aligned to slides and filtered."""
    resp = ctx.response
    if resp.graph.nodes:
        raw = [(n.id, n.label, n.description, n.frequency) for n in resp.graph.nodes]
    else:
        raw = [(normalize(c.name), c.name, None, c.frequency) for c in resp.concepts]
    seen: set[str] = set()
    out: list[ConceptRecord] = []
    for cid, label, desc, freq in raw:
        if cid in seen:
            continue
        seen.add(cid)
        idx = locate_slides(label, cid, materials.slides)
        out.append(
            ConceptRecord(
                id=cid,
                label=label,
                description=desc,
                frequency=freq,
                slide_indexes=idx,
                alignment_status=AlignmentStatus.ALIGNED if idx else AlignmentStatus.NOT_FOUND,
            )
        )
    return (concept_filter or MockConceptFilter()).apply(out)
