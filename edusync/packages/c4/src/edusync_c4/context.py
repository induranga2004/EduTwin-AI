"""Align annotation groups with speech, slide text and concept names (evidence only).

Nothing here interprets meaning: it collects what was said, shown and drawn near a group and
records where each piece came from. Concept matches are *candidates* (a name appears in the
text), never proof of coverage or understanding.
"""

import re
from typing import Protocol

from pydantic import BaseModel, Field

from edusync_c4.schemas import (
    AnnotationContext,
    AnnotationGroup,
    AnnotationTimeline,
    ConceptCandidate,
    ConceptHit,
    NormalizedAnnotation,
    SlideText,
    SpeechEvidence,
    SpeechSegment,
)
from edusync_core import ConceptRef


class ContextConfig(BaseModel):
    speech_pad_seconds: float = Field(default=90.0, ge=0.0)
    # Added to transcript times to express them on the annotation (lecture-relative) clock.
    speech_offset_seconds: float = 0.0
    max_slide_excerpt_chars: int = Field(default=500, ge=1)


class ConceptMatcher(Protocol):
    def match(self, text: str, concepts: list[ConceptRef]) -> list[ConceptHit]: ...


class ExactPhraseMatcher:
    """Case-insensitive whole-word phrase match of each concept label."""

    method = "exact-phrase"

    def match(self, text: str, concepts: list[ConceptRef]) -> list[ConceptHit]:
        hits: list[ConceptHit] = []
        for concept in concepts:
            label = concept.label.strip()
            if not label:
                continue
            found = re.search(r"(?<!\w)" + re.escape(label) + r"(?!\w)", text, re.IGNORECASE)
            if found:
                hits.append(ConceptHit(concept=concept, matched_text=found.group(0)))
        return hits


def members_of(group: AnnotationGroup, timeline: AnnotationTimeline) -> list[NormalizedAnnotation]:
    """Latest placed, non-deleted version of every shape in the group (for recognizers)."""
    wanted = set(group.shape_ids)
    latest: dict[str, NormalizedAnnotation] = {}
    for ev in timeline.events:
        if ev.shape_id in wanted and not ev.is_deleted and not ev.is_clear:
            latest[ev.shape_id] = ev
    return [latest[s] for s in group.shape_ids if s in latest]


def _relation(seg_start: float, seg_end: float, span_start: float, span_end: float) -> str:
    if seg_end < span_start:
        return "before"
    if seg_start > span_end:
        return "after"
    return "during"


def build_group_context(
    group: AnnotationGroup,
    speech: list[SpeechSegment],
    slide_text: SlideText | None,
    concepts: list[ConceptRef],
    config: ContextConfig | None = None,
    matcher: ConceptMatcher | None = None,
    diagram_sentences: list[str] | None = None,
) -> AnnotationContext:
    """Collect speech, slide text and concept candidates around one group.

    Concept candidates come from the group's own native text and nearby speech only.
    """
    cfg = config or ContextConfig()
    matcher = matcher or ExactPhraseMatcher()
    method = getattr(matcher, "method", type(matcher).__name__)

    window_start = group.start_seconds - cfg.speech_pad_seconds
    window_end = group.end_seconds + cfg.speech_pad_seconds

    speech_ev: list[SpeechEvidence] = []
    for seg in sorted(speech, key=lambda s: (s.start, str(s.id))):
        start = seg.start + cfg.speech_offset_seconds
        end = seg.end + cfg.speech_offset_seconds
        if end < window_start or start > window_end:
            continue
        speech_ev.append(
            SpeechEvidence(
                evidence_id=f"speech:{seg.id}",
                segment_id=seg.id,
                start=start,
                end=end,
                text=seg.text.strip(),
                relation=_relation(start, end, group.start_seconds, group.end_seconds),
            )
        )

    excerpt: str | None = None
    truncated = False
    slide_eid: str | None = None
    if slide_text is not None and slide_text.slide_id == group.slide_id and slide_text.text.strip():
        full = slide_text.text.strip()
        slide_eid = f"slide:{slide_text.slide_id}"
        truncated = len(full) > cfg.max_slide_excerpt_chars
        excerpt = full[: cfg.max_slide_excerpt_chars].rstrip() if truncated else full

    sources: list[tuple[str, str, str]] = []
    if group.native_text:
        sources.append(("annotation", f"annotation:{group.group_id}", group.native_text))
    # Slide text is whole-slide (no coordinates), so concept names found in it would attach to
    # every group on the slide. It is kept as context only, not as a concept source.
    sources.extend(("speech", s.evidence_id, s.text) for s in speech_ev)

    candidates: list[ConceptCandidate] = []
    for kind, eid, text in sources:
        for hit in matcher.match(text, concepts):
            candidates.append(
                ConceptCandidate(
                    concept=hit.concept,
                    matched_text=hit.matched_text,
                    source=kind,
                    evidence_id=eid,
                    method=method,
                )
            )

    return AnnotationContext(
        group_id=group.group_id,
        lecture_id=group.lecture_id,
        slide_id=group.slide_id,
        span_start=group.start_seconds,
        span_end=group.end_seconds,
        window_start=window_start,
        window_end=window_end,
        native_text=group.native_text,
        slide_evidence_id=slide_eid,
        slide_text_excerpt=excerpt,
        slide_text_truncated=truncated,
        speech=speech_ev,
        concept_candidates=candidates,
        diagram_sentences=list(diagram_sentences or []),
    )
