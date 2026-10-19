"""Rule-based concept coverage tracking over timestamped speech segments.

A rule-based baseline, not the trained research classifier; mentions are not mastery.
"""

import re
from collections import defaultdict

from pydantic import BaseModel

from edusync_c1.schemas import ConceptEdge, KnowledgeGraph, TranscriptSegment
from edusync_core.models import ConceptRef, EvidenceItem, EvidenceSource, LectureTimestamp


class SegmentEvidence(BaseModel):
    """One speech segment in which a concept was mentioned."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    text: str


class CoverageNode(BaseModel):
    id: str
    label: str
    status: str
    evidence_count: int
    first_seen_seconds: float | None = None
    latest_seen_seconds: float | None = None


class CoverageSnapshot(BaseModel):
    """Graph state after one segment."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    transcript_text: str
    matched_concepts: list[str]
    nodes: list[CoverageNode]
    edges: list[ConceptEdge]


def _status(count: int) -> str:
    if count == 0:
        return "not_yet_observed"
    return "introduced" if count == 1 else "repeatedly_mentioned"


class CoverageTracker:
    """Tracks concept evidence as timestamped lecture segments arrive."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self.edges = graph.edges
        self.labels: dict[str, str] = {}
        self._patterns: dict[str, re.Pattern[str]] = {}
        self.evidence: dict[str, list[SegmentEvidence]] = defaultdict(list)
        for node in graph.nodes:
            label = node.label.strip()
            if node.id and label:
                self.labels[node.id] = node.label
                self._patterns[node.id] = re.compile(
                    r"(?<!\w)" + re.escape(label) + r"(?!\w)", re.IGNORECASE
                )

    def process_segment(self, segment: TranscriptSegment) -> CoverageSnapshot:
        """Update concept evidence using one speech segment."""
        text = segment.text.strip()
        matched: list[str] = []
        if text:
            for concept_id, pattern in self._patterns.items():
                if pattern.search(text):
                    self.evidence[concept_id].append(
                        SegmentEvidence(
                            segment_id=segment.id,
                            start_seconds=segment.start,
                            end_seconds=segment.end,
                            text=text,
                        )
                    )
                    matched.append(concept_id)

        nodes = []
        for concept_id, label in self.labels.items():
            items = self.evidence[concept_id]
            nodes.append(
                CoverageNode(
                    id=concept_id,
                    label=label,
                    status=_status(len(items)),
                    evidence_count=len(items),
                    first_seen_seconds=items[0].start_seconds if items else None,
                    latest_seen_seconds=items[-1].start_seconds if items else None,
                )
            )
        return CoverageSnapshot(
            segment_id=segment.id,
            start_seconds=segment.start,
            end_seconds=segment.end,
            transcript_text=text,
            matched_concepts=matched,
            nodes=nodes,
            edges=self.edges,
        )

    def process_segments(self, segments: list[TranscriptSegment]) -> list[CoverageSnapshot]:
        """Process segments in chronological order."""
        return [self.process_segment(s) for s in sorted(segments, key=lambda s: s.start)]

    def evidence_items(self, meeting_id: str) -> list[EvidenceItem]:
        """Export recorded mentions as core ``EvidenceItem``s (source=SPEECH)."""
        items = [
            EvidenceItem(
                concept=ConceptRef(id=concept_id, label=self.labels[concept_id]),
                source=EvidenceSource.SPEECH,
                timestamp=LectureTimestamp(meeting_id=meeting_id, seconds=ev.start_seconds),
                text=ev.text,
                meta={"segment_id": ev.segment_id, "end_seconds": ev.end_seconds},
            )
            for concept_id, evs in self.evidence.items()
            for ev in evs
        ]
        return sorted(items, key=lambda i: (i.timestamp.seconds, i.concept.id))
