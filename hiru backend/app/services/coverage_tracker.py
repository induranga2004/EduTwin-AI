
import re
from collections import defaultdict


class CoverageTracker:
    """
    Tracks concept evidence as timestamped lecture segments arrive.

    This is a rule-based prototype, not the trained research classifier.
    """

    def __init__(self, graph: dict):
        self.nodes = graph.get("nodes", [])
        self.edges = graph.get("edges", [])

        self.concepts = {}
        self.evidence = defaultdict(list)

        for node in self.nodes:
            concept_id = node.get("id")
            label = node.get("label", "")

            if concept_id and label.strip():
                self.concepts[concept_id] = {
                    "id": concept_id,
                    "label": label,
                    "pattern": re.compile(
                        r"(?<!\w)"
                        + re.escape(label.strip())
                        + r"(?!\w)",
                        re.IGNORECASE,
                    ),
                }

    def process_segment(self, segment: dict) -> dict:
        """Update concept evidence using one timestamped speech segment."""

        text = segment.get("text", "").strip()
        start = float(segment.get("start_seconds", 0))
        end = float(segment.get("end_seconds", start))
        segment_id = segment.get("segment_id")

        matched_concepts = []

        if text:
            for concept_id, concept in self.concepts.items():
                if concept["pattern"].search(text):
                    evidence_item = {
                        "segment_id": segment_id,
                        "start_seconds": start,
                        "end_seconds": end,
                        "text": text,
                    }

                    self.evidence[concept_id].append(evidence_item)
                    matched_concepts.append(concept_id)

        updated_nodes = []

        for concept_id, concept in self.concepts.items():
            evidence_items = self.evidence[concept_id]
            count = len(evidence_items)

            if count == 0:
                status = "not_yet_observed"
            elif count == 1:
                status = "introduced"
            else:
                status = "repeatedly_mentioned"

            updated_nodes.append({
                "id": concept_id,
                "label": concept["label"],
                "status": status,
                "evidence_count": count,
                "first_seen_seconds": (
                    evidence_items[0]["start_seconds"]
                    if evidence_items else None
                ),
                "latest_seen_seconds": (
                    evidence_items[-1]["start_seconds"]
                    if evidence_items else None
                ),
            })

        return {
            "segment_id": segment_id,
            "start_seconds": start,
            "end_seconds": end,
            "transcript_text": text,
            "matched_concepts": matched_concepts,
            "nodes": updated_nodes,
            "edges": self.edges,
        }

    def process_segments(self, segments: list[dict]) -> list[dict]:
        """Process segments in chronological order."""

        ordered_segments = sorted(
            segments,
            key=lambda item: float(item.get("start_seconds", 0)),
        )

        return [
            self.process_segment(segment)
            for segment in ordered_segments
        ]
