"""Topological Late-Joiner Catch-Up Engine.

Generates a mathematically ordered, 5-minute fast-track catch-up pack with audio replay
anchors and rapid self-check quizzes for students joining a lecture late.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from edusync_c1.schemas import KnowledgeGraph, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)


class AudioAnchor(BaseModel):
    """Timestamped audio playback window for reviewing a missed concept."""

    start_seconds: float
    end_seconds: float
    duration_seconds: float
    preview_text: str


class CatchupConceptSummary(BaseModel):
    """Pedagogically ordered summary of a missed concept."""

    concept_id: str
    concept_label: str
    chain_depth: int
    prerequisites: list[str]
    takeaway_sentence: str
    audio_anchor: AudioAnchor
    estimated_review_seconds: int = 45


class CatchupQuizItem(BaseModel):
    """Rapid self-check question to verify prerequisite comprehension."""

    id: str
    concept_id: str
    concept_label: str
    question: str
    options: list[str]
    correct_option_index: int
    explanation: str


class LateJoinerCatchupPack(BaseModel):
    """Personalized 5-minute fast-track catch-up bundle for late-arriving students."""

    lecture_id: str
    join_timestamp_seconds: float
    missed_concept_count: int
    estimated_catchup_minutes: float
    ordered_concepts: list[CatchupConceptSummary]
    quiz: list[CatchupQuizItem]
    note: str


class LateJoinerCatchupEngine:
    """Computes topologically ordered catch-up summaries using the curriculum DAG."""

    def __init__(self, planned_graph: KnowledgeGraph) -> None:
        self.planned_graph = planned_graph
        self.nodes_by_id = {n.id: n for n in planned_graph.nodes}
        self.labels_by_id = {n.id: n.label for n in planned_graph.nodes}

        # Build NetworkX DiGraph for topological traversal
        self.dag = nx.DiGraph()
        for node in planned_graph.nodes:
            self.dag.add_node(node.id, label=node.label, description=node.description)

        for edge in planned_graph.edges:
            if edge.relation_type == "prerequisite":
                self.dag.add_edge(edge.source, edge.target, confidence=edge.confidence)

    def compute_chain_depth(self, concept_id: str) -> int:
        """Calculate topological depth (longest prerequisite ancestor chain length)."""
        ancestors = nx.ancestors(self.dag, concept_id) if concept_id in self.dag else set()
        if not ancestors:
            return 0
        sub = self.dag.subgraph(ancestors | {concept_id})
        return nx.dag_longest_path_length(sub)

    def generate_catchup_pack(
        self,
        delivered_mentions: list[tuple[float, list[str], str]],  # (timestamp, concept_ids, text)
        join_timestamp_seconds: float,
        lecture_id: str = "iisa-lecture-01",
        max_review_seconds_per_concept: int = 45,
    ) -> LateJoinerCatchupPack:
        """Construct the 5-minute topologically ordered catch-up pack."""
        # 1. Filter mentions delivered prior to student arrival
        missed_evidence: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for ts, cids, text in delivered_mentions:
            if ts <= join_timestamp_seconds:
                for cid in cids:
                    if cid in self.nodes_by_id:
                        missed_evidence[cid].append((ts, text))

        missed_concept_ids = set(missed_evidence.keys())
        if not missed_concept_ids:
            return LateJoinerCatchupPack(
                lecture_id=lecture_id,
                join_timestamp_seconds=round(join_timestamp_seconds, 2),
                missed_concept_count=0,
                estimated_catchup_minutes=0.0,
                ordered_concepts=[],
                quiz=[],
                note="No concepts were missed prior to your join time.",
            )

        # 2. Induce Subgraph and Perform Pedagogical Topological Sort
        subgraph = self.dag.subgraph(missed_concept_ids).copy()

        # In case of any cycle or skipped intermediary, ensure foundational depth ordering
        if not nx.is_directed_acyclic_graph(subgraph):
            topological_order = sorted(
                list(missed_concept_ids),
                key=lambda cid: (self.compute_chain_depth(cid), cid),
            )
        else:
            topo_gen = list(nx.topological_generations(subgraph))
            topological_order = []
            for gen in topo_gen:
                topological_order.extend(
                    sorted(gen, key=lambda cid: (self.compute_chain_depth(cid), cid))
                )

        # 3. Build Concept Summaries & Audio Anchors
        summaries: list[CatchupConceptSummary] = []
        quiz_items: list[CatchupQuizItem] = []

        for cid in topological_order:
            node = self.nodes_by_id[cid]
            evidence_list = missed_evidence[cid]
            depth = self.compute_chain_depth(cid)

            # Direct prerequisite parent labels
            parents = [
                self.labels_by_id[u]
                for u in self.dag.predecessors(cid)
                if u in self.labels_by_id
            ]

            # Best audio replay anchor: window around the earliest or longest explanation
            earliest_ts, anchor_text = min(evidence_list, key=lambda e: e[0])
            start_sec = max(0.0, earliest_ts - 5.0)
            end_sec = start_sec + float(max_review_seconds_per_concept)
            audio_anchor = AudioAnchor(
                start_seconds=round(start_sec, 2),
                end_seconds=round(end_sec, 2),
                duration_seconds=round(end_sec - start_sec, 2),
                preview_text=anchor_text[:140] + ("..." if len(anchor_text) > 140 else ""),
            )

            takeaway = node.description if node.description else anchor_text[:160]

            summaries.append(
                CatchupConceptSummary(
                    concept_id=cid,
                    concept_label=node.label,
                    chain_depth=depth,
                    prerequisites=parents,
                    takeaway_sentence=takeaway,
                    audio_anchor=audio_anchor,
                    estimated_review_seconds=max_review_seconds_per_concept,
                )
            )

            # Generate rapid self-check question
            quiz_items.append(
                CatchupQuizItem(
                    id=f"quiz-{cid}",
                    concept_id=cid,
                    concept_label=node.label,
                    question=f"Which statement correctly defines {node.label}?",
                    options=[
                        takeaway,
                        f"{node.label} is an optional user interface optimization without security implications.",
                        f"{node.label} is strictly deprecated and superseded by non-cryptographic protocols.",
                    ],
                    correct_option_index=0,
                    explanation=f"Key takeaway: {takeaway}",
                )
            )

        total_review_sec = len(summaries) * max_review_seconds_per_concept
        catchup_minutes = round(total_review_sec / 60.0, 1)

        note = (
            f"Topologically ordered {len(summaries)} missed concepts in prerequisite sequence. "
            f"Review audio clips in sequence ({catchup_minutes} min total) to catch up seamlessly."
        )

        return LateJoinerCatchupPack(
            lecture_id=lecture_id,
            join_timestamp_seconds=round(join_timestamp_seconds, 2),
            missed_concept_count=len(summaries),
            estimated_catchup_minutes=catchup_minutes,
            ordered_concepts=summaries,
            quiz=quiz_items,
            note=note,
        )
