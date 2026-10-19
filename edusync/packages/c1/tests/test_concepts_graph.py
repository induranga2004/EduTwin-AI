import pytest

from edusync_c1.concepts import extract_concepts
from edusync_c1.graph import build_knowledge_graph
from edusync_c1.schemas import Concept, Page


@pytest.mark.parametrize(("text", "limit"), [("", 30), ("   ", 30), ("Some Text Here", 0)])
def test_extract_concepts_empty(text, limit):
    assert extract_concepts(text, limit) == []


def test_extract_concepts_ranks_by_frequency():
    text = "Gradient Descent is key. Gradient Descent converges. Loss Function helps. the and of"
    concepts = extract_concepts(text)
    assert concepts[0].name == "Gradient Descent"
    assert concepts[0].frequency >= 2
    assert all(c.name.casefold() not in {"the", "and", "of"} for c in concepts)


def test_extract_concepts_dedups_case_and_limits():
    concepts = extract_concepts("neural network and Neural Network and neural network", 2)
    names = [c.name.casefold() for c in concepts]
    assert len(names) == len(set(names)) <= 2


def test_graph_builds_co_mention_edges():
    pages = [
        Page(page=1, text="Loss Function drives Gradient Descent. Learning Rate is separate."),
        Page(page=2, text="Gradient Descent and Loss Function again.\n\n"),
    ]
    concepts = [
        Concept(name="Gradient Descent", frequency=2),
        Concept(name="Loss Function", frequency=2),
        Concept(name="Learning Rate", frequency=1),
        Concept(name="gradient descent", frequency=9),  # duplicate key ignored
        Concept(name="  ", frequency=1),
    ]
    graph = build_knowledge_graph(pages, concepts)
    assert graph.node_count == 3
    assert graph.edge_count == 1
    edge = graph.edges[0]
    assert (edge.source, edge.target) == ("gradient descent", "loss function")
    assert edge.evidence_count == 2
    assert edge.relation_type == "co_mentioned"
    assert "not verified prerequisite" in graph.note


def test_graph_limits_edges_and_evidence():
    page = Page(page=1, text=" ".join("Alpha Beta Gamma." for _ in range(5)))
    concepts = [Concept(name=n) for n in ("Alpha", "Beta", "Gamma")]
    full = build_knowledge_graph([page], concepts)
    assert full.edge_count == 3
    assert all(len(e.evidence) == 3 and e.evidence_count == 5 for e in full.edges)
    assert build_knowledge_graph([page], concepts, max_edges=1).edge_count == 1
    assert build_knowledge_graph([page], concepts, max_edges=-5).edge_count == 0


def test_graph_empty_inputs():
    graph = build_knowledge_graph([], [])
    assert graph.nodes == []
    assert graph.edges == []
