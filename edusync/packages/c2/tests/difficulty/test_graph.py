"""GraphView: valid prerequisite edges, metrics, cycles and ignored edges."""

import random

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


def _edge_keys(view: GraphView, ids: list[str]) -> list[tuple[str, str, float | None, str | None]]:
    return [
        (e.source, e.target, e.confidence, e.source_type)
        for i in ids
        for e in view.prerequisites_of(i)
    ]


def test_default_arguments_are_identical() -> None:
    edges = [_pre("a", "b"), _pre("a", "b"), _pre("a", "ghost"), _pre("b", "c", confidence=0.4)]
    resp = _graph(["a", "b", "c"], edges)
    old, new = GraphView(resp), GraphView(resp, aliases={}, dropped=())
    assert old.warnings == new.warnings
    assert _edge_keys(old, ["a", "b", "c"]) == _edge_keys(new, ["a", "b", "c"])
    assert old.metrics("b") == new.metrics("b")
    m = old.metrics("b")
    assert m is not None
    assert m.direct_count == 2
    for v in (old, new):
        assert (v.merged_aliases, v.rewritten_edges, v.dropped_edges, v.merge_self_loops) == (
            0,
            0,
            0,
            0,
        )


def test_v0_unaffected_by_aliases(graph_v0: C1GraphResponse) -> None:
    view = GraphView(graph_v0, aliases={"x": "y"}, dropped=["nothing"])
    assert view.has_prerequisite_edges is False
    assert view.warnings == []


def test_alias_remap() -> None:
    view = GraphView(_graph(["a", "b", "b2"], [_pre("a", "b2")]), aliases={"b2": "b"})
    assert (view.merged_aliases, view.rewritten_edges) == (1, 1)
    assert [(e.source, e.target) for e in view.prerequisites_of("b")] == [("a", "b")]
    assert view.prerequisites_of("b2") == view.prerequisites_of("b")
    assert view.node("b2") is view.node("b")
    node = view.node("b")
    assert node is not None
    assert node.id == "b"
    m = view.metrics("b2")
    assert m is not None
    assert m == view.metrics("b")
    assert m.direct_count == 1


def test_alias_nodes_are_not_exposed() -> None:
    view = GraphView(_graph(["a", "b"], []), aliases={"b": "a"}, dropped=["zzz"])
    assert set(view._nodes) == {"a"}


def test_flattened_chain_aliases() -> None:
    # a=b and b=c flattened by the caller to a->c, b->c; an unflattened map is followed one hop.
    resp = _graph(["a", "b", "c", "d"], [_pre("a", "d"), _pre("b", "d")])
    view = GraphView(resp, aliases={"a": "c", "b": "c"})
    edges = view.prerequisites_of("d")
    assert [(e.source, e.target) for e in edges] == [("c", "d")]
    one_hop = GraphView(resp, aliases={"a": "b", "b": "c"})
    assert sorted(e.source for e in one_hop.prerequisites_of("d")) == ["b", "c"]


def test_diamond_with_alias() -> None:
    edges = [_pre("a", "b"), _pre("a", "c2"), _pre("b", "d"), _pre("c", "d")]
    view = GraphView(_graph(["a", "b", "c", "c2", "d"], edges), aliases={"c2": "c"})
    m = view.metrics("d")
    assert m is not None
    assert (m.direct_count, m.chain_depth, m.transitive_count, m.has_cycle) == (2, 2, 3, False)


def test_duplicate_precedence() -> None:
    ids = ["a", "a2", "b"]
    amap = {"a2": "a"}

    def kept(edges: list[dict]) -> tuple[float | None, str | None]:
        out = GraphView(_graph(ids, edges), aliases=amap).prerequisites_of("b")
        assert len(out) == 1
        return out[0].confidence, out[0].source_type

    assert kept([_pre("a", "b", confidence=0.9), _pre("a2", "b", source_type="inferred")]) == (
        None,
        "inferred",
    )
    assert kept(
        [
            _pre("a", "b", source_type="inferred", confidence=0.9),
            _pre("a2", "b", source_type="curriculum_defined"),
        ]
    ) == (None, "curriculum_defined")
    assert kept(
        [
            _pre("a", "b", source_type="inferred", confidence=0.3),
            _pre("a2", "b", source_type="inferred", confidence=0.6),
        ]
    ) == (0.6, "inferred")
    assert kept(
        [
            _pre("a", "b", source_type="inferred"),
            _pre("a2", "b", source_type="inferred", confidence=0.1),
        ]
    ) == (0.1, "inferred")
    assert kept([_pre("a", "b", confidence=0.5), _pre("a2", "b", confidence=0.5)]) == (0.5, None)
    first = GraphView(
        _graph(ids, [_pre("a2", "b", evidence_count=1), _pre("a", "b", evidence_count=2)]),
        aliases=amap,
    ).prerequisites_of("b")
    assert first[0].evidence_count == 1
    assert first[0].source == "a"


def test_merge_self_loops_warn_once() -> None:
    edges = [_pre("a", "a2"), _pre("a2", "a"), _pre("a", "a")]
    view = GraphView(_graph(["a", "a2"], edges), aliases={"a2": "a"})
    assert view.merge_self_loops == 2
    assert view.has_prerequisite_edges is False
    assert view.warnings[0].endswith("self-loop")  # the raw self-loop keeps its own warning
    assert view.warnings[1:] == [
        "2 prerequisite edge(s) became self-loops after alias merge and were ignored"
    ]


def test_dropped_cuts_chain() -> None:
    edges = [_pre("a", "b"), _pre("b", "c"), _pre("c", "d")]
    view = GraphView(_graph(["a", "b", "c", "d"], edges), dropped=["b"])
    assert view.dropped_edges == 2
    assert view.node("b") is None
    assert view.metrics("b") is None
    m = view.metrics("d")
    assert m is not None
    assert (m.direct_count, m.chain_depth, m.transitive_count) == (1, 1, 1)
    c = view.metrics("c")
    assert c is not None
    assert c.direct_count == 0


def test_dropped_canonical_removes_alias_edges() -> None:
    view = GraphView(
        _graph(["a", "b", "b2"], [_pre("a", "b2")]), aliases={"b2": "b"}, dropped=["b"]
    )
    assert view.dropped_edges == 1
    assert view.has_prerequisite_edges is False
    assert view.metrics("b2") is None


def test_alias_on_both_sides_of_edges() -> None:
    edges = [_pre("x2", "y"), _pre("y", "z2"), _pre("x", "z")]
    view = GraphView(_graph(["x", "x2", "y", "z", "z2"], edges), aliases={"x2": "x", "z2": "z"})
    assert (view.rewritten_edges, view.merged_aliases) == (2, 2)
    m = view.metrics("z")
    assert m is not None
    assert m.direct_count == 2
    assert (m.chain_depth, m.transitive_count) == (2, 2)


def test_merge_created_cycle_terminates() -> None:
    # a -> b and b2 -> a2 with a2=a, b2=b gives a 2-cycle a <-> b after merging.
    edges = [_pre("a", "b"), _pre("b2", "a2")]
    view = GraphView(_graph(["a", "a2", "b", "b2"], edges), aliases={"a2": "a", "b2": "b"})
    for cid in ("a", "b"):
        m = view.metrics(cid)
        assert m is not None
        assert m.has_cycle is True
        assert m.direct_count == 1


def test_unknown_alias_entries_are_ignored() -> None:
    view = GraphView(
        _graph(["a", "b"], [_pre("a", "b")]), aliases={"a": "ghost", "ghost": "b", "b": "b"}
    )
    assert view.merged_aliases == 0
    assert view.rewritten_edges == 0
    assert view.node("a") is not None
    assert [(e.source, e.target) for e in view.prerequisites_of("b")] == [("a", "b")]


def test_order_invariance() -> None:
    nodes = ["a", "a2", "b", "c", "c2"]
    edges = [
        _pre("a", "b", source_type="inferred", confidence=0.2),
        _pre("a2", "b", source_type="curriculum_defined", confidence=0.1),
        _pre("b", "c", confidence=0.4),
        _pre("b", "c2", confidence=0.8),
        _pre("c2", "a2", confidence=0.3),
    ]
    amap = {"a2": "a", "c2": "c"}
    ref = GraphView(_graph(nodes, edges), aliases=amap)
    ids = ["a", "b", "c"]
    expected = {i: ref.metrics(i) for i in ids}
    bm = expected["b"]
    assert bm is not None
    assert bm.has_cycle is True
    rng = random.Random(7)
    for _ in range(10):
        n2, e2 = nodes[:], edges[:]
        rng.shuffle(n2)
        rng.shuffle(e2)
        v = GraphView(_graph(n2, e2), aliases=amap)
        assert {i: v.metrics(i) for i in ids} == expected
        kept = v.prerequisites_of("b")
        assert [(e.source_type, e.confidence) for e in kept] == [("curriculum_defined", 0.1)]


# --------------------------------------------------------- dependents_count and weighted_depth


def _m(view: GraphView, cid: str):
    m = view.metrics(cid)
    assert m is not None
    return m


def test_v0_dependents_and_weighted_depth_are_zero(graph_v0: C1GraphResponse) -> None:
    view = GraphView(graph_v0)
    for node in graph_v0.graph.nodes:
        assert _m(view, node.id).dependents_count == 0
        assert _m(view, node.id).weighted_depth == 0.0
        assert view.dependents_of(node.id) == []


def test_v1_dependents_and_weighted_depth(graph_v1: C1GraphResponse) -> None:
    view = GraphView(graph_v1)
    assert {n.id: _m(view, n.id).dependents_count for n in graph_v1.graph.nodes} == {
        "loss function": 1,  # the co_mentioned edge does not count
        "learning rate": 1,
        "gradient descent": 1,
        "backpropagation": 0,
    }
    assert _m(view, "gradient descent").weighted_depth == pytest.approx(0.9)
    assert _m(view, "backpropagation").weighted_depth == pytest.approx(1.6)
    assert _m(view, "loss function").weighted_depth == 0.0
    assert view.dependents_of("learning rate") == ["gradient descent"]
    assert view.dependents_of("nope") == []


def test_unknown_concept_has_no_metrics_or_dependents() -> None:
    view = GraphView(_graph(["a", "b"], [_pre("a", "b")]))
    assert view.metrics("nope") is None
    assert view.dependents_of("nope") == []


def test_diamond_weighted_depth_takes_the_heaviest_chain() -> None:
    edges = [
        _pre("a", "b", confidence=0.9),
        _pre("a", "c", confidence=0.1),
        _pre("b", "d", confidence=0.5),
        _pre("c", "d", confidence=0.8),
    ]
    view = GraphView(_graph(["a", "b", "c", "d"], edges))
    assert _m(view, "d").weighted_depth == pytest.approx(1.4)  # a->b->d beats a->c->d (0.9)
    assert _m(view, "d").chain_depth == 2
    assert _m(view, "a").dependents_count == 2
    assert _m(view, "b").dependents_count == 1
    assert _m(view, "d").dependents_count == 0


def test_missing_confidence_counts_as_one() -> None:
    edges = [_pre("a", "b"), _pre("b", "c", confidence=0.25)]
    view = GraphView(_graph(["a", "b", "c"], edges))
    assert _m(view, "b").weighted_depth == 1.0
    assert _m(view, "c").weighted_depth == pytest.approx(1.25)


def test_cycle_weighted_depth_terminates_and_stays_finite() -> None:
    edges = [_pre("b", "a", confidence=0.5), _pre("c", "b", confidence=0.5), _pre("a", "c")]
    view = GraphView(_graph(["a", "b", "c"], edges))
    m = _m(view, "a")
    assert m.has_cycle is True
    # a <- b <- c, the edge a -> c would revisit a and is ignored: 0.5 + 0.5
    assert m.weighted_depth == pytest.approx(1.0)
    assert (m.dependents_count, m.chain_depth) == (1, 2)


def test_cycle_memo_does_not_leak_between_queries() -> None:
    edges = [_pre("a", "b", confidence=0.5), _pre("b", "a", confidence=0.5), _pre("b", "c")]
    view = GraphView(_graph(["a", "b", "c"], edges))
    first = _m(view, "c")
    assert _m(view, "c") == first
    assert first.has_cycle is True


def test_aliases_merge_dependents_and_dedupe() -> None:
    edges = [
        _pre("a", "b", confidence=0.3),
        _pre("a2", "b", confidence=0.6),  # duplicate of a -> b after the merge
        _pre("a2", "c", confidence=0.2),
    ]
    view = GraphView(_graph(["a", "a2", "b", "c"], edges), aliases={"a2": "a"})
    assert view.dependents_of("a2") == ["b", "c"]
    assert _m(view, "a").dependents_count == 2
    assert _m(view, "b").weighted_depth == pytest.approx(0.6)  # the better duplicate is kept


def test_dropped_concepts_are_cut_from_dependents() -> None:
    edges = [_pre("a", "b"), _pre("a", "c"), _pre("c", "d")]
    view = GraphView(_graph(["a", "b", "c", "d"], edges), dropped={"c"})
    assert _m(view, "a").dependents_count == 1
    assert view.metrics("c") is None
    assert _m(view, "d").weighted_depth == 0.0


def test_dependents_and_weighted_depth_are_order_invariant() -> None:
    nodes = ["a", "a2", "b", "c", "d"]
    edges = [
        _pre("a", "b", confidence=0.2),
        _pre("a2", "b", confidence=0.7),
        _pre("b", "c", confidence=0.4),
        _pre("a", "c"),
        _pre("c", "d", confidence=0.9),
        _pre("d", "a2", confidence=0.1),
    ]
    amap = {"a2": "a"}
    ids = ["a", "b", "c", "d"]
    ref = GraphView(_graph(nodes, edges), aliases=amap)
    expected = {i: (_m(ref, i).dependents_count, _m(ref, i).weighted_depth) for i in ids}
    rng = random.Random(11)
    for _ in range(10):
        n2, e2 = nodes[:], edges[:]
        rng.shuffle(n2)
        rng.shuffle(e2)
        v = GraphView(_graph(n2, e2), aliases=amap)
        assert {i: (_m(v, i).dependents_count, _m(v, i).weighted_depth) for i in ids} == expected
        assert {i: sorted(v.dependents_of(i)) for i in ids} == {
            i: sorted(ref.dependents_of(i)) for i in ids
        }
