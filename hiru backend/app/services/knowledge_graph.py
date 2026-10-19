import re
from itertools import combinations
import networkx as nx


def build_knowledge_graph(
    pages: list[dict],
    concepts: list[dict],
    max_edges: int = 200,
) -> dict:
    """
    Build a document-evidence graph.

    Nodes are extracted concepts. Edges mean two concepts co-occur in a
    sentence; they do NOT establish prerequisite relationships.
    """
    graph = nx.Graph()
    concept_lookup = {}

    for concept in concepts or []:
        name = str(concept.get("name", "")).strip()
        if not name:
            continue
        key = name.casefold()
        if key not in concept_lookup:
            try:
                frequency = int(concept.get("frequency", 0) or 0)
            except (TypeError, ValueError):
                frequency = 0
            concept_lookup[key] = {"name": name, "frequency": frequency}
            graph.add_node(key, label=name, frequency=frequency)

    edge_evidence = {}
    for page in pages or []:
        page_number = page.get("page")
        text = str(page.get("text", "") or "")
        if not text:
            continue
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.casefold()
            mentioned = []
            for key in concept_lookup:
                pattern = r"(?<!\w)" + re.escape(key) + r"(?!\w)"
                if re.search(pattern, lower):
                    mentioned.append(key)
            for source, target in combinations(sorted(set(mentioned)), 2):
                pair = (source, target)
                evidence = edge_evidence.setdefault(pair, {"count": 0, "evidence": []})
                evidence["count"] += 1
                if len(evidence["evidence"]) < 3:
                    evidence["evidence"].append({
                        "page": page_number,
                        "sentence": sentence[:500],
                    })

    try:
        edge_limit = max(0, int(max_edges))
    except (TypeError, ValueError):
        edge_limit = 200

    ranked = sorted(edge_evidence.items(), key=lambda item: (-item[1]["count"], item[0]))
    for (source, target), evidence in ranked[:edge_limit]:
        graph.add_edge(
            source,
            target,
            relation_type="co_mentioned",
            evidence_count=evidence["count"],
            evidence=evidence["evidence"],
        )

    nodes = [
        {"id": node_id, "label": attrs["label"], "frequency": attrs["frequency"]}
        for node_id, attrs in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": source,
            "target": target,
            "relation_type": attrs["relation_type"],
            "evidence_count": attrs["evidence_count"],
            "evidence": attrs["evidence"],
        }
        for source, target, attrs in graph.edges(data=True)
    ]
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "note": (
            "Edges indicate concepts mentioned together in the document. "
            "They are not verified prerequisite relationships."
        ),
    }