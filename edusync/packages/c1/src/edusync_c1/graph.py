"""Document-evidence knowledge graph built from extracted concepts."""

import re
from itertools import combinations

import networkx as nx

from edusync_c1.schemas import (
    Concept,
    ConceptEdge,
    ConceptNode,
    EdgeEvidence,
    KnowledgeGraph,
    Page,
)

GRAPH_NOTE = (
    "Edges indicate concepts mentioned together in the document. "
    "They are not verified prerequisite relationships."
)
_MAX_EVIDENCE_PER_EDGE = 3


def build_knowledge_graph(
    pages: list[Page], concepts: list[Concept], max_edges: int = 200
) -> KnowledgeGraph:
    """Build a graph whose nodes are concepts and whose edges mean sentence co-occurrence."""
    graph = nx.Graph()
    patterns: dict[str, re.Pattern[str]] = {}
    for concept in concepts:
        name = concept.name.strip()
        key = name.casefold()
        if name and key not in patterns:
            patterns[key] = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)")
            graph.add_node(key, label=name, frequency=concept.frequency)

    # (source, target) -> [count, evidence]
    pair_evidence: dict[tuple[str, str], tuple[int, list[EdgeEvidence]]] = {}
    for page in pages:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", page.text):
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.casefold()
            mentioned = sorted(key for key, pat in patterns.items() if pat.search(lower))
            for pair in combinations(mentioned, 2):
                count, evidence = pair_evidence.get(pair, (0, []))
                if len(evidence) < _MAX_EVIDENCE_PER_EDGE:
                    evidence.append(EdgeEvidence(page=page.page, sentence=sentence[:500]))
                pair_evidence[pair] = (count + 1, evidence)

    ranked = sorted(pair_evidence.items(), key=lambda item: (-item[1][0], item[0]))
    for (source, target), (count, evidence) in ranked[: max(0, max_edges)]:
        graph.add_edge(source, target, evidence_count=count, evidence=evidence)

    nodes = [
        ConceptNode(id=node_id, label=attrs["label"], frequency=attrs["frequency"])
        for node_id, attrs in graph.nodes(data=True)
    ]
    edges = [
        ConceptEdge(
            source=source,
            target=target,
            evidence_count=attrs["evidence_count"],
            evidence=attrs["evidence"],
        )
        for source, target, attrs in graph.edges(data=True)
    ]
    return KnowledgeGraph(
        node_count=len(nodes),
        edge_count=len(edges),
        nodes=nodes,
        edges=edges,
        note=GRAPH_NOTE,
    )
