"""Concept coverage tracking over timestamped speech segments.

Supports both rule-based tracking and Model C supervised coverage classification
combining MiniLM semantic similarity, prerequisite parent completion ratios, and evidence counts.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from edusync_c1.schemas import ConceptEdge, KnowledgeGraph, TranscriptSegment
from edusync_c1.semantic import SemanticMatcher
from edusync_core.models import ConceptRef, EvidenceItem, EvidenceSource, LectureTimestamp

logger = logging.getLogger(__name__)


class SegmentEvidence(BaseModel):
    """One speech segment in which a concept was mentioned."""

    segment_id: int
    start_seconds: float
    end_seconds: float
    text: str
    similarity: float = 1.0


class CoverageNode(BaseModel):
    """Cumulative coverage state of a concept node."""

    id: str
    label: str
    status: str
    evidence_count: int
    first_seen_seconds: float | None = None
    latest_seen_seconds: float | None = None
    confidence: float = 1.0
    prereq_coverage_ratio: float = 1.0


class CoverageSnapshot(BaseModel):
    """Graph coverage state after processing one speech segment."""

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


def resolve_classifier_path(candidate: str | Path | None = None) -> Path | None:
    """Find the path to the trained Model C classifier artifact."""
    if candidate and Path(candidate).is_file():
        return Path(candidate)

    pkg_models = Path(__file__).resolve().parents[2] / "models"
    for name in ("coverage_classifier.joblib", "coverage_classifier_minilm.joblib"):
        p = pkg_models / name
        if p.is_file():
            return p
    return None


def load_coverage_classifier(path: Path | None = None) -> Any:
    """Load the joblib serialized Model C classifier pipeline."""
    target = path or resolve_classifier_path()
    if target and target.is_file():
        try:
            import sys
            import joblib

            try:
                import edusync_c1.research.train_model as tm

                sys.modules["__main__"].MiniLMTextEmbedder = tm.MiniLMTextEmbedder
            except Exception:
                pass

            return joblib.load(target)
        except Exception:
            logger.warning("Could not load coverage classifier from %s", target)
    return None


class CoverageTracker:
    """Tracks concept evidence as timestamped lecture segments arrive.

    Supports both exact regex keyword matching and MiniLM semantic bi-encoder matching,
    and can optionally evaluate Model C to predict calibrated coverage status.
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        *,
        matcher: SemanticMatcher | None = None,
        classifier: Any = None,
        use_ml_classifier: bool = False,
    ) -> None:
        self.edges = graph.edges
        self.labels: dict[str, str] = {}
        self.canonical_names: list[str] = []
        self._patterns: dict[str, re.Pattern[str]] = {}
        self.evidence: dict[str, list[SegmentEvidence]] = defaultdict(list)
        self.matcher = matcher
        self.classifier = classifier
        self.use_ml_classifier = use_ml_classifier

        if self.use_ml_classifier and self.classifier is None:
            self.classifier = load_coverage_classifier()

        # Build prerequisite parent graph: target -> list of source prerequisite IDs
        self.prerequisites: dict[str, list[str]] = defaultdict(list)
        for edge in graph.edges:
            if edge.relation_type == "prerequisite":
                self.prerequisites[edge.target].append(edge.source)

        for node in graph.nodes:
            label = node.label.strip()
            if node.id and label:
                self.labels[node.id] = node.label
                self.canonical_names.append(node.label)
                self._patterns[node.id] = re.compile(
                    r"(?<!\w)" + re.escape(label) + r"(?!\w)", re.IGNORECASE
                )

    def _calculate_prereq_ratio(self, concept_id: str) -> float:
        """Proportion of prerequisite parent concepts that have recorded evidence."""
        parents = self.prerequisites.get(concept_id, [])
        if not parents:
            return 1.0
        covered = sum(1 for p in parents if len(self.evidence[p]) > 0)
        return round(covered / len(parents), 4)

    def process_segment(self, segment: TranscriptSegment) -> CoverageSnapshot:
        """Update concept evidence using one speech segment."""
        text = segment.text.strip()
        matched: list[str] = []
        matched_scores: dict[str, float] = {}

        if text:
            # 1. MiniLM Semantic Matching (if matcher provided)
            if self.matcher is not None:
                match_res = self.matcher.match(text, self.canonical_names)
                label_to_id = {v.casefold(): k for k, v in self.labels.items()}
                for m in match_res.matches:
                    cid = label_to_id.get(m.name.casefold())
                    if cid and cid not in matched:
                        matched.append(cid)
                        matched_scores[cid] = m.similarity
                        self.evidence[cid].append(
                            SegmentEvidence(
                                segment_id=segment.id,
                                start_seconds=segment.start,
                                end_seconds=segment.end,
                                text=text,
                                similarity=m.similarity,
                            )
                        )
            else:
                # 2. Rule-based exact regex matching
                for concept_id, pattern in self._patterns.items():
                    if pattern.search(text):
                        self.evidence[concept_id].append(
                            SegmentEvidence(
                                segment_id=segment.id,
                                start_seconds=segment.start,
                                end_seconds=segment.end,
                                text=text,
                                similarity=1.0,
                            )
                        )
                        matched.append(concept_id)
                        matched_scores[concept_id] = 1.0

        nodes = []
        for concept_id, label in self.labels.items():
            items = self.evidence[concept_id]
            count = len(items)
            prereq_ratio = self._calculate_prereq_ratio(concept_id)

            status_label = _status(count)
            confidence = 1.0

            # 3. Model C ML Classifier Prediction (if enabled and loaded)
            if self.use_ml_classifier and self.classifier is not None and count > 0:
                try:
                    import pandas as pd

                    latest_text = items[-1].text if items else text
                    feat_df = pd.DataFrame(
                        [
                            {
                                "transcript_text": latest_text,
                                "prereq_coverage_ratio": prereq_ratio,
                                "evidence_count": count,
                            }
                        ]
                    )
                    pred = self.classifier.predict(feat_df)[0]
                    status_label = str(pred)
                    if hasattr(self.classifier, "predict_proba"):
                        proba = self.classifier.predict_proba(feat_df)[0]
                        confidence = round(float(max(proba)), 3)
                except Exception:
                    logger.debug("Model C inference fallback to rule-based status.")
                    status_label = _status(count)

            nodes.append(
                CoverageNode(
                    id=concept_id,
                    label=label,
                    status=status_label,
                    evidence_count=count,
                    first_seen_seconds=items[0].start_seconds if items else None,
                    latest_seen_seconds=items[-1].start_seconds if items else None,
                    confidence=confidence,
                    prereq_coverage_ratio=prereq_ratio,
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
