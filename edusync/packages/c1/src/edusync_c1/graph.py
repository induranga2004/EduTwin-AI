"""Document-evidence knowledge graph built from extracted concepts.

Combines deterministic slide temporal order and linguistic dependency cues with local MiniLM
sentence-transformer embeddings to construct a directed prerequisite DAG.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations
from typing import Any

import networkx as nx

from edusync_c1.schemas import (
    Concept,
    ConceptEdge,
    ConceptNode,
    EdgeEvidence,
    KnowledgeGraph,
    Page,
)
from edusync_c1.semantic import SemanticMatcher

logger = logging.getLogger(__name__)

GRAPH_NOTE = (
    "Edges indicate concepts mentioned together in the document. "
    "They are not verified prerequisite relationships."
)
DAG_NOTE = (
    "Curriculum prerequisite knowledge graph generated using temporal slide precedence, "
    "linguistic dependency heuristics, and MiniLM semantic embeddings. Verified acyclic DAG."
)
_MAX_EVIDENCE_PER_EDGE = 3
DEFAULT_MIN_SEMANTIC_AFFINITY = 0.25

_DEFINITION_KEYWORDS = re.compile(
    r"\b(?:is defined as|refers to|is a|are|provides|consists of|utilizes|represents|focuses on|describes)\b",
    re.IGNORECASE,
)

_PREREQ_LINGUISTIC_PATTERNS = [
    re.compile(
        r"\b(?P<target>[a-z0-9_\s]{3,40}?)\s+(?:requires|builds on|depends on|is a prerequisite for)\s+(?P<source>[a-z0-9_\s]{3,40})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<source>[a-z0-9_\s]{3,40}?)\s+(?:is required for|is a foundation for|must be studied before|is needed for)\s+(?P<target>[a-z0-9_\s]{3,40})\b",
        re.IGNORECASE,
    ),
]


def _truncate_sentence(sentence: str, max_chars: int = 200) -> str:
    cleaned = " ".join(sentence.split())
    if len(cleaned) <= max_chars:
        return cleaned
    truncated = cleaned[:max_chars].rsplit(" ", 1)[0]
    return truncated.rstrip(",;:- ") + "..."


def extract_concept_description(
    concept_name: str,
    sentences: list[str],
    matcher: SemanticMatcher | None = None,
) -> str:
    """Extract or rank the most informative definition sentence for a concept."""
    if not sentences:
        return ""

    candidates = [s.strip() for s in sentences if len(s.strip()) > 15]
    if not candidates:
        candidates = [s.strip() for s in sentences if s.strip()]
    if not candidates:
        return ""

    # 1. MiniLM Neural Ranking: embed candidates and pick highest cosine similarity
    if matcher is not None:
        try:
            model = matcher._get_model()
            if model is not None and len(candidates) > 1:
                eval_pool = candidates[:12]
                embeddings = model.encode(
                    [concept_name, *eval_pool],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                scores = embeddings[1:] @ embeddings[0]
                best_idx = int(scores.argmax())
                return _truncate_sentence(eval_pool[best_idx])
        except Exception:
            logger.debug("MiniLM ranking failed in description extraction; using heuristic fallback.")

    # 2. Heuristic fallback: pick sentence matching definitional keywords
    for s in candidates:
        if _DEFINITION_KEYWORDS.search(s):
            return _truncate_sentence(s)

    # 3. Default to the first sentence where concept was introduced
    return _truncate_sentence(candidates[0])


def _check_linguistic_dependency(
    sentence: str, key_a: str, key_b: str
) -> tuple[str, str, float] | None:
    """Detect explicit grammatical prerequisite cues between key_a and key_b.

    Returns (source, target, confidence) where source is prerequisite of target, or None.
    """
    lower = sentence.casefold()
    for pattern in _PREREQ_LINGUISTIC_PATTERNS:
        match = pattern.search(lower)
        if not match:
            continue
        matched_source = match.group("source").strip()
        matched_target = match.group("target").strip()
        if key_a in matched_source and key_b in matched_target:
            return (key_a, key_b, 0.95)
        if key_b in matched_source and key_a in matched_target:
            return (key_b, key_a, 0.95)
    return None


def build_knowledge_graph(
    pages: list[Page],
    concepts: list[Concept],
    max_edges: int = 200,
    *,
    matcher: SemanticMatcher | None = None,
    infer_prerequisites: bool = True,
    min_semantic_affinity: float = DEFAULT_MIN_SEMANTIC_AFFINITY,
) -> KnowledgeGraph:
    """Build an enriched curriculum knowledge graph from lecture pages and concepts.

    Extracts concept descriptions, infers directed prerequisite edges using temporal slide order
    and linguistic cues, computes MiniLM semantic relatedness, and eliminates cycles to guarantee
    an acyclic Directed Acyclic Graph (DAG).
    """
    if not concepts:
        return KnowledgeGraph(node_count=0, edge_count=0, nodes=[], edges=[], note=GRAPH_NOTE)

    patterns: dict[str, re.Pattern[str]] = {}
    canonical_labels: dict[str, str] = {}
    concept_frequencies: dict[str, int] = {}
    concept_sentences: dict[str, list[tuple[int, str]]] = {}
    first_seen_pages: dict[str, int] = {}

    for concept in concepts:
        name = concept.name.strip()
        key = name.casefold()
        if name and key not in patterns:
            patterns[key] = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)")
            canonical_labels[key] = name
            concept_frequencies[key] = concept.frequency
            concept_sentences[key] = []
            first_seen_pages[key] = 999999

    # Scan pages: record first seen page, candidate sentences, and co-occurrences
    pair_evidence: dict[tuple[str, str], tuple[int, list[EdgeEvidence], list[str]]] = {}
    for page in pages:
        page_num = page.page or 1
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.casefold()
            mentioned = sorted(key for key, pat in patterns.items() if pat.search(lower))
            for key in mentioned:
                concept_sentences[key].append((page_num, sentence))
                if page_num < first_seen_pages[key]:
                    first_seen_pages[key] = page_num

            for pair in combinations(mentioned, 2):
                count, evidence, raw_sentences = pair_evidence.get(pair, (0, [], []))
                if len(evidence) < _MAX_EVIDENCE_PER_EDGE:
                    evidence.append(EdgeEvidence(page=page_num, sentence=sentence[:500]))
                raw_sentences.append(sentence)
                pair_evidence[pair] = (count + 1, evidence, raw_sentences)

    # 1. Build Concept Nodes (enriched with descriptions prioritized from introductory slide)
    nodes: list[ConceptNode] = []
    for key, label in canonical_labels.items():
        all_sentences = concept_sentences.get(key, [])
        first_page = first_seen_pages.get(key, 999999)
        intro_sentences = [s for p, s in all_sentences if p == first_page]
        chosen_candidates = intro_sentences if intro_sentences else [s for _, s in all_sentences]
        desc = extract_concept_description(label, chosen_candidates, matcher=matcher)
        nodes.append(
            ConceptNode(
                id=key,
                label=label,
                frequency=concept_frequencies.get(key, 0),
                description=desc,
            )
        )

    # 2. Infer Directed Prerequisite Edges & Co-occurrences
    candidate_edges: list[ConceptEdge] = []
    ranked_pairs = sorted(pair_evidence.items(), key=lambda item: (-item[1][0], item[0]))

    for (k1, k2), (count, evidence, raw_sentences) in ranked_pairs:
        # Check explicit linguistic cues across sentences
        ling_match = None
        for s in raw_sentences:
            ling_match = _check_linguistic_dependency(s, k1, k2)
            if ling_match:
                break

        if ling_match:
            src, tgt, conf = ling_match
            candidate_edges.append(
                ConceptEdge(
                    source=src,
                    target=tgt,
                    relation_type="prerequisite",
                    direction="source_is_prerequisite_of_target",
                    confidence=conf,
                    source_type="linguistic_dependency",
                    evidence_count=count,
                    evidence=evidence,
                )
            )
            continue

        p1 = first_seen_pages.get(k1, 999999)
        p2 = first_seen_pages.get(k2, 999999)

        if infer_prerequisites and p1 != p2:
            # Temporal slide precedence: earlier concept is prerequisite of later concept
            src, tgt = (k1, k2) if p1 < p2 else (k2, k1)
            edge_conf = 0.75

            # MiniLM Neural Affinity Weighting
            if matcher is not None:
                try:
                    model = matcher._get_model()
                    if model is not None:
                        src_name = canonical_labels[src]
                        tgt_name = canonical_labels[tgt]
                        embs = model.encode(
                            [src_name, tgt_name],
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        )
                        affinity = float(embs[0] @ embs[1])
                        if affinity < min_semantic_affinity:
                            # Below affinity threshold: retain as co_mentioned without prerequisite arrow
                            candidate_edges.append(
                                ConceptEdge(
                                    source=min(k1, k2),
                                    target=max(k1, k2),
                                    relation_type="co_mentioned",
                                    direction="undirected",
                                    confidence=round(max(0.1, affinity), 3),
                                    source_type="low_semantic_affinity",
                                    evidence_count=count,
                                    evidence=evidence,
                                )
                            )
                            continue
                        edge_conf = round(max(0.5, affinity), 3)
                except Exception:
                    logger.debug("MiniLM affinity scoring failed; retaining temporal default confidence.")

            candidate_edges.append(
                ConceptEdge(
                    source=src,
                    target=tgt,
                    relation_type="prerequisite",
                    direction="source_is_prerequisite_of_target",
                    confidence=edge_conf,
                    source_type="temporal_slide_precedence",
                    evidence_count=count,
                    evidence=evidence,
                )
            )
        else:
            # Same-slide co-occurrence or without temporal precedence
            candidate_edges.append(
                ConceptEdge(
                    source=min(k1, k2),
                    target=max(k1, k2),
                    relation_type="co_mentioned",
                    direction="undirected",
                    confidence=0.5,
                    source_type="sentence_co_occurrence",
                    evidence_count=count,
                    evidence=evidence,
                )
            )

    # 3. DAG Cycle Elimination with NetworkX
    di_graph = nx.DiGraph()
    for node in nodes:
        di_graph.add_node(node.id)

    prereq_edges: list[ConceptEdge] = []
    other_edges: list[ConceptEdge] = []
    for edge in candidate_edges:
        if edge.relation_type == "prerequisite":
            di_graph.add_edge(
                edge.source,
                edge.target,
                confidence=edge.confidence,
                edge_obj=edge,
            )
        else:
            other_edges.append(edge)

    while not nx.is_directed_acyclic_graph(di_graph):
        cycles = list(nx.simple_cycles(di_graph))
        if not cycles:
            break
        cycle = cycles[0]
        cycle_edges = list(zip(cycle, cycle[1:] + [cycle[0]]))
        min_u, min_v = min(
            cycle_edges,
            key=lambda e: di_graph[e[0]][e[1]].get("confidence", 0.0),
        )
        removed_edge_data = di_graph[min_u][min_v]
        di_graph.remove_edge(min_u, min_v)

        # Convert broken cycle edge to undirected co_mentioned so evidence is preserved
        broken_edge: ConceptEdge = removed_edge_data["edge_obj"]
        broken_edge.relation_type = "co_mentioned"
        broken_edge.direction = "undirected"
        broken_edge.source_type = "cycle_broken_co_occurrence"
        other_edges.append(broken_edge)

    # Collect surviving DAG prerequisite edges
    for u, v, data in di_graph.edges(data=True):
        prereq_edges.append(data["edge_obj"])

    final_edges = sorted(
        prereq_edges + other_edges,
        key=lambda e: (-e.evidence_count, e.source, e.target),
    )

    if max_edges >= 0:
        final_edges = final_edges[:max_edges]
    else:
        final_edges = []

    has_co_mentioned = any(e.relation_type == "co_mentioned" for e in final_edges)
    has_prereq = any(e.relation_type == "prerequisite" for e in final_edges)

    if has_prereq:
        note = (
            f"{DAG_NOTE} "
            f"Edges without directional evidence are retained as co_mentioned (not verified prerequisite relationships)."
        )
    else:
        note = GRAPH_NOTE

    return KnowledgeGraph(
        node_count=len(nodes),
        edge_count=len(final_edges),
        nodes=nodes,
        edges=final_edges,
        note=note,
    )
