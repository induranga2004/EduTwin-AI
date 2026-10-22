"""Dual-graph comparison engine for real-time curriculum alignment and skip alerts.

Computes the graph differential Δ(t) = G_planned \\ G_delivered(t) to identify
prerequisite violations where advanced concepts are taught before their foundational
prerequisites have been delivered.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from edusync_c1.schemas import ConceptEdge, KnowledgeGraph

logger = logging.getLogger(__name__)


class PrerequisiteSkipAlert(BaseModel):
    """Alert emitted when an advanced concept is delivered before its prerequisites."""

    timestamp_seconds: float
    concept_id: str
    concept_label: str
    skipped_prerequisites: list[str]
    skipped_prerequisite_labels: list[str]
    severity: str = "warning"  # "warning" | "critical"
    message: str


class GraphDifferentialSnapshot(BaseModel):
    """Differential state Δ(t) at a specific lecture timestamp."""

    timestamp_seconds: float
    delivered_concept_ids: list[str]
    pending_concept_ids: list[str]
    coverage_ratio: float
    alignment_score: float
    active_alerts: list[PrerequisiteSkipAlert] = Field(default_factory=list)


class LectureComparisonReport(BaseModel):
    """Full lecture curriculum alignment evaluation report."""

    lecture_id: str = "lecture-comparison"
    total_planned_concepts: int
    total_delivered_concepts: int
    coverage_ratio: float
    pedagogical_alignment_score: float
    total_skip_alerts: int
    alerts: list[PrerequisiteSkipAlert]
    differential_timeline: list[GraphDifferentialSnapshot]


class DualGraphComparator:
    """Compares the static planned curriculum DAG against dynamic delivered speech evidence."""

    def __init__(self, planned_graph: KnowledgeGraph) -> None:
        self.planned_graph = planned_graph
        self.node_labels: dict[str, str] = {n.id: n.label for n in planned_graph.nodes}
        self.all_node_ids: set[str] = set(self.node_labels.keys())

        # Build prerequisite parent mapping and directed DAG
        self.prerequisites: dict[str, list[str]] = defaultdict(list)
        self.dag = nx.DiGraph()
        for n in planned_graph.nodes:
            self.dag.add_node(n.id)
        for edge in planned_graph.edges:
            if edge.relation_type == "prerequisite":
                self.prerequisites[edge.target].append(edge.source)
                self.dag.add_edge(edge.source, edge.target)

    def get_prerequisite_ratio(self, concept_id: str, delivered_ids: set[str]) -> float:
        """Calculate the proportion of completed prerequisites for a given concept."""
        reqs = self.prerequisites.get(concept_id, [])
        if not reqs:
            return 1.0
        met = sum(1 for r in reqs if r in delivered_ids)
        return round(met / len(reqs), 4)

    def evaluate_concept_mention(
        self,
        concept_id: str,
        delivered_ids: set[str],
        timestamp_seconds: float,
    ) -> PrerequisiteSkipAlert | None:
        """Check if introducing concept_id at timestamp_seconds violates prerequisite ordering."""
        # Check all upstream prerequisite ancestors in the sequential DAG
        if concept_id in self.dag:
            ancestors = nx.ancestors(self.dag, concept_id)
        else:
            ancestors = set(self.prerequisites.get(concept_id, []))

        # Filter out the overarching root main concept from slide skip alert
        root_key = self.planned_graph.main_concept
        if root_key:
            ancestors.discard(root_key)

        if not ancestors:
            return None

        # Order missing prerequisites according to the planned graph nodes sequence
        missing = [
            n.id for n in self.planned_graph.nodes
            if n.id in ancestors and n.id not in delivered_ids
        ]
        if not missing:
            return None

        missing_labels = [self.node_labels.get(r, r) for r in missing]
        label = self.node_labels.get(concept_id, concept_id)
        minutes = int(timestamp_seconds // 60)
        seconds = int(timestamp_seconds % 60)

        severity = "critical" if len(missing) > 1 else "warning"
        msg = (
            f"Prerequisite Skip Alert ({severity.upper()}): Concept '{label}' was introduced at "
            f"{minutes:02d}:{seconds:02d} without prior delivery of foundational prerequisite(s): "
            f"{', '.join(missing_labels)}."
        )

        return PrerequisiteSkipAlert(
            timestamp_seconds=round(timestamp_seconds, 2),
            concept_id=concept_id,
            concept_label=label,
            skipped_prerequisites=missing,
            skipped_prerequisite_labels=missing_labels,
            severity=severity,
            message=msg,
        )

    def compare_timeline_steps(
        self,
        step_mentions: list[tuple[float, list[str]]],
        lecture_id: str = "iisa-lecture",
    ) -> LectureComparisonReport:
        """Evaluate a chronological sequence of (timestamp_seconds, [mentioned_concept_ids])."""
        delivered: set[str] = set()
        alerts: list[PrerequisiteSkipAlert] = []
        snapshots: list[GraphDifferentialSnapshot] = []

        total_nodes = len(self.all_node_ids) or 1
        total_prereq_edges = sum(len(reqs) for reqs in self.prerequisites.values()) or 1
        violation_count = 0

        for ts, mentioned in step_mentions:
            step_alerts: list[PrerequisiteSkipAlert] = []
            for cid in mentioned:
                if cid not in self.all_node_ids:
                    continue
                # If this concept hasn't been delivered yet, evaluate prerequisite compliance
                if cid not in delivered:
                    alert = self.evaluate_concept_mention(cid, delivered, ts)
                    if alert:
                        step_alerts.append(alert)
                        alerts.append(alert)
                        violation_count += len(alert.skipped_prerequisites)
                    delivered.add(cid)

            pending = sorted(list(self.all_node_ids - delivered))
            cov_ratio = round(len(delivered) / total_nodes, 4)
            alignment = round(max(0.0, 1.0 - (violation_count / total_prereq_edges)), 4)

            snapshots.append(
                GraphDifferentialSnapshot(
                    timestamp_seconds=round(ts, 2),
                    delivered_concept_ids=sorted(list(delivered)),
                    pending_concept_ids=pending,
                    coverage_ratio=cov_ratio,
                    alignment_score=alignment,
                    active_alerts=step_alerts,
                )
            )

        final_cov = round(len(delivered) / total_nodes, 4)
        final_alignment = round(max(0.0, 1.0 - (violation_count / total_prereq_edges)), 4)

        return LectureComparisonReport(
            lecture_id=lecture_id,
            total_planned_concepts=len(self.all_node_ids),
            total_delivered_concepts=len(delivered),
            coverage_ratio=final_cov,
            pedagogical_alignment_score=final_alignment,
            total_skip_alerts=len(alerts),
            alerts=alerts,
            differential_timeline=snapshots,
        )
