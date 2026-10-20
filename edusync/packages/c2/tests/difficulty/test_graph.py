"""GraphView: valid prerequisite edges, metrics, cycles and ignored edges."""

import pytest

from edusync_c2.difficulty.graph import GraphView
from edusync_c2.difficulty.schemas import C1GraphResponse


def _graph(ids: list[str], edges: list[dict]) -> C1GraphResponse:
    return C1GraphResponse.model_validate(
        {
            "graph": {
                "nodes": [{"id": i, "label": i.upper()} for i in ids],
                "edges": edges,
            }
        }
    )


def _pre(s: str, t: str, **extra: object) -> dict:
    return {"source": s, "target": t, "relation_type": "prerequisite", **extra}


def test_v0_has_no_prerequisites_but_metrics_are_zero(graph_v0: C1GraphResponse) -> None:
    view = GraphView(graph_v0)
    assert view.has_prerequisite_edges is False
    assert view.warnings == []
    for node in graph_v0.graph.nodes:
        m = view.metrics(node.id)
        assert m is not None
        assert (m.direct_count, m.chain_depth, m.transitive_count) == (0, 0, 0)
        assert m.mean_confidence is None
        assert m.curriculum_defined_share is None
        assert m.has_cycle is False


def test_v1_chains_and_aggregates(graph_v1: C1GraphResponse) -> None:
    view = GraphView(graph_v1)
    assert view.has_prerequisite_edges is True
    gd = view.metrics("gradient descent")
    assert gd is not None
    assert gd.direct_count == 2
    assert gd.chain_depth == 1
    assert gd.transitive_count == 2
    assert gd.mean_confidence == pytest.approx(0.75)
    assert gd.curriculum_defined_share == pytest.approx(0.5)
    bp = view.metrics("backpropagation")
    assert bp is not None
    assert (bp.direct_count, bp.chain_depth, bp.transitive_count) == (1, 2, 3)
    assert bp.curriculum_defined_share == 0.0
    # co_mentioned is never a prerequisite
    assert view.prerequisites_of("loss function") == []
    assert [e.source for e in view.prerequisites_of("gradient descent")] == [
        "loss function",
        "learning rate",
    ]


def test_unknown_node(graph_v1: C1GraphResponse) -> None:
    view = GraphView(graph_v1)
    assert view.metrics("nope") is None
    assert view.node("nope") is None
    assert view.node("learning rate") is not None
    assert view.prerequisites_of("nope") == []


def test_cycle_is_safe() -> None:
    view = GraphView(_graph(["a", "b", "c"], [_pre("b", "a"), _pre("c", "b"), _pre("a", "c")]))
    m = view.metrics("a")
    assert m is not None
    assert m.has_cycle is True
    assert m.chain_depth == 2
    assert m.transitive_count == 2


def test_diamond_depth_uses_cache() -> None:
    edges = [_pre("a", "b"), _pre("a", "c"), _pre("b", "d"), _pre("c", "d")]
    m = GraphView(_graph(["a", "b", "c", "d"], edges)).metrics("d")
    assert m is not None
    assert (m.direct_count, m.chain_depth, m.transitive_count, m.has_cycle) == (2, 2, 3, False)


def test_invalid_edges_are_ignored_with_deduplicated_warnings() -> None:
    edges = [
        _pre("a", "b", direction="target_is_prerequisite_of_source"),
        _pre("a", "b", direction="target_is_prerequisite_of_source"),
        _pre("a", "ghost"),
        _pre("a", "a"),
        {"source": "a", "target": "b", "relation_type": "related_to"},
        {"source": "a", "target": "b", "relation_type": "co_mentioned"},
    ]
    view = GraphView(_graph(["a", "b"], edges))
    assert view.has_prerequisite_edges is False
    assert len(view.warnings) == 3
    assert "direction" in view.warnings[0]
    assert "unknown node" in view.warnings[1]
    assert "self-loop" in view.warnings[2]
    assert GraphView(_graph(["a", "b"], edges)).warnings == view.warnings


def test_edge_without_confidence() -> None:
    view = GraphView(_graph(["a", "b"], [_pre("a", "b")]))
    m = view.metrics("b")
    assert m is not None
    assert m.mean_confidence is None
    assert m.curriculum_defined_share == 0.0
