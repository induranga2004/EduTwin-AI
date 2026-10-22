"""Transcript normalisation and cumulative graph timeline."""

from typing import Any

from pydantic import BaseModel

from edusync_c1.concepts import extract_concepts
from edusync_c1.documents import combine_page_text
from edusync_c1.graph import build_knowledge_graph
from edusync_c1.schemas import (
    GraphTimeline,
    MatchingInfo,
    Page,
    TimelineGraph,
    TimelineNode,
    TimelineStep,
    Transcript,
    TranscriptSegment,
)
from edusync_c1.semantic import DEFAULT_SIMILARITY_THRESHOLD, MODEL_NAME, SemanticMatcher

TIMELINE_NOTE = (
    "Semantic matches are candidate evidence that a concept may be mentioned. The threshold "
    "must be calibrated using labelled examples. Mention does not prove adequate coverage, "
    "student understanding, mastery, or a prerequisite relationship."
)


def _as_dict(obj: Any) -> dict[str, Any]:
    """View a dict, pydantic model or plain object as a dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    raise ValueError("The transcription service returned an unsupported result.")


def normalize_transcription_result(result: Any) -> Transcript:
    """Normalise a dict / pydantic / object transcription result into a ``Transcript``."""
    data = _as_dict(result)
    segments = []
    for index, raw in enumerate(data.get("segments") or []):
        seg = _as_dict(raw)
        segments.append(
            TranscriptSegment(
                id=seg.get("id", index),
                start=float(seg.get("start", 0) or 0),
                end=float(seg.get("end", 0) or 0),
                text=str(seg.get("text", "")).strip(),
            )
        )
    text = str(data.get("text", "")).strip() or " ".join(s.text for s in segments if s.text)
    return Transcript(segments=segments, language=data.get("language"), text=text)


def build_timeline_compatibility(
    transcript: Transcript,
    pages: list[Page],
    matcher: SemanticMatcher | None = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> GraphTimeline:
    """Build cumulative graph snapshots by matching transcript segments to concepts.

    A match is evidence of a possible mention, not proof of mastery or full coverage.
    """
    matcher = matcher or SemanticMatcher()
    concepts = extract_concepts(combine_page_text(pages))
    graph = build_knowledge_graph(pages, concepts)
    counts = {node.id: 0 for node in graph.nodes}

    steps: list[TimelineStep] = []
    for segment in transcript.segments:
        start = segment.start
        end = segment.end or start
        result = matcher.match(segment.text, concepts, threshold=similarity_threshold)
        scores = {m.name.casefold(): m.similarity for m in result.matches}

        for node in graph.nodes:
            if node.id in scores:
                counts[node.id] += 1

        # Synthesize coverage for the main concept when its constituent slide concepts are covered
        main_id = graph.main_concept
        if main_id and main_id in counts:
            constituent_nodes = [n for n in graph.nodes if n.id != main_id]
            if constituent_nodes:
                covered_constituents = sum(1 for n in constituent_nodes if counts[n.id] > 0)
                if (covered_constituents / len(constituent_nodes) >= 0.5) or (
                    len(constituent_nodes) >= 4 and covered_constituents >= 3
                ):
                    if counts[main_id] == 0:
                        counts[main_id] = covered_constituents

        nodes = []
        for node in graph.nodes:
            status = "not_yet_observed"
            if counts[node.id] > 0:
                status = "adequately_covered" if (counts[node.id] >= 2 or node.id == main_id) else "introduced"
            nodes.append(
                TimelineNode(
                    **node.model_dump(),
                    evidence_count=counts[node.id],
                    status=status,
                    latest_similarity=scores.get(node.id),
                )
            )
        steps.append(
            TimelineStep(
                segment_id=segment.id,
                start_seconds=start,
                end_seconds=end,
                transcript_text=segment.text,
                matched_concepts=[m.name for m in result.matches],
                concept_matches=result.matches,
                matching_method=result.method,
                nodes=nodes,
                edges=graph.edges,
            )
        )

    initial = TimelineGraph(
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        nodes=[TimelineNode(**n.model_dump()) for n in graph.nodes],
        edges=graph.edges,
        note=graph.note,
    )
    return GraphTimeline(
        initial_graph=initial,
        timeline=steps,
        matching=MatchingInfo(
            preferred_method="MiniLM cosine similarity",
            model=MODEL_NAME,
            similarity_threshold=similarity_threshold,
            fallback="case-insensitive exact phrase matching",
        ),
        note=TIMELINE_NOTE,
    )
