
import re
from itertools import combinations

import networkx as nx


def build_knowledge_graph(
    pages: list[dict],
    concepts: list[dict],
    max_edges: int = 200,
) -> dict:
    """
    Build a candidate knowledge graph from extracted lecture concepts.

    Nodes represent candidate concepts.
    Edges represent concepts mentioned together in a sentence.

    Co-mention is not proof of a prerequisite relationship.
    Relationships must be validated before being treated as prerequisites.
    """

    graph = nx.Graph()

    # Add the concepts extracted by our concept extractor.
    concept_lookup = {}

    for concept in concepts:
        name = concept.get("name", "").strip()

        if not name:
            continue

        key = name.casefold()

        if key not in concept_lookup:
            concept_lookup[key] = {
                "name": name,
                "frequency": concept.get("frequency", 0),
            }

    # Add a node for each candidate concept.
    for key, concept in concept_lookup.items():
        graph.add_node(
            key,
            label=concept["name"],
            frequency=concept["frequency"],
        )

    # Identify sentences that mention multiple concepts.
    edge_evidence = {}

    for page in pages:
        page_number = page.get("page")
        text = page.get("text", "")

        if not text:
            continue

        sentences = re.split(
            r"(?<=[.!?])\s+|\n+",
            text,
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_lower = sentence.casefold()
            mentioned = []

            for key, concept in concept_lookup.items():
                pattern = (
                    r"(?<!\w)"
                    + re.escape(key)
                    + r"(?!\w)"
                )

                if re.search(pattern, sentence_lower):
                    mentioned.append(key)

            # Avoid creating an edge from a sentence containing
            # only one concept.
            for source, target in combinations(
                sorted(set(mentioned)), 2
            ):
                pair = (source, target)

                if pair not in edge_evidence:
                    edge_evidence[pair] = {
                        "count": 0,
                        "evidence": [],
                    }

                evidence = edge_evidence[pair]
                evidence["count"] += 1

                # Retain a small number of supporting sentences.
                if len(evidence["evidence"]) < 3:
                    evidence["evidence"].append(
                        {
                            "page": page_number,
                            "sentence": sentence[:500],
                        }
                    )

    # Rank candidate edges by their co-mention frequency.
    ranked_edges = sorted(
        edge_evidence.items(),
        key=lambda item: (
            -item[1]["count"],
            item[0],
        ),
    )

    for (source, target), evidence in ranked_edges[:max_edges]:
        graph.add_edge(
            source,
            target,
            relation_type="co_mentioned",
            evidence_count=evidence["count"],
            evidence=evidence["evidence"],
        )

    # Convert the graph into JSON-friendly data for the API.
    nodes = [
        {
            "id": node_id,
            "label": attributes["label"],
            "frequency": attributes["frequency"],
        }
        for node_id, attributes in graph.nodes(data=True)
    ]

    edges = [
        {
            "source": source,
            "target": target,
            "relation_type": attributes["relation_type"],
            "evidence_count": attributes["evidence_count"],
            "evidence": attributes["evidence"],
        }
        for source, target, attributes in graph.edges(data=True)
    ]

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "note": (
            "Edges indicate concepts mentioned together in the text. "
            "They are not verified prerequisite relationships."
        ),
    }
