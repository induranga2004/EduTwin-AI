"""Tests for the C1 graph contract checker."""

import copy
from collections.abc import Callable

import pytest

from edusync_c2.difficulty.contract_check import RULES, ContractReport, check_c1_graph
from edusync_c2.difficulty.schemas import C1GraphResponse
from edusync_c2.difficulty.synthetic import build_c1_graph, make_deck_spec


def slugify_ids(data: dict) -> dict:
    """Return a copy of a graph with hyphenated slug ids, as contract v1 expects."""
    out = copy.deepcopy(data)

    def slug(value: str) -> str:
        return "-".join(value.split())

    for node in out["graph"]["nodes"]:
        node["id"] = slug(node["id"])
    for edge in out["graph"]["edges"]:
        edge["source"], edge["target"] = slug(edge["source"]), slug(edge["target"])
    return out


def rules_of(report: ContractReport) -> set[str]:
    return {f.rule for f in report.findings}


@pytest.fixture
def v0(graph_v0: C1GraphResponse) -> dict:
    return graph_v0.model_dump(mode="json")


@pytest.fixture
def v1(graph_v1: C1GraphResponse) -> dict:
    return slugify_ids(graph_v1.model_dump(mode="json"))


def test_rule_table_is_unique_and_described() -> None:
    ids = [r[0] for r in RULES]
    assert len(ids) == len(set(ids))
    assert all(r[2].endswith(".") for r in RULES)


def test_valid_v0_and_v1(v0: dict, v1: dict) -> None:
    r0, r1 = check_c1_graph(v0), check_c1_graph(v1)
    assert (r0.version, r0.ok, r0.findings) == ("v0", True, [])
    assert r0.stats == {
        "nodes": 4,
        "edges": 1,
        "prerequisite_edges": 0,
        "co_mentioned_edges": 1,
        "errors": 0,
        "warnings": 0,
    }
    assert (r1.version, r1.ok) == ("v1", True)
    assert r1.stats["prerequisite_edges"] == 3
    assert r1.stats["co_mentioned_edges"] == 1


@pytest.mark.parametrize("version", ["v0", "v1"])
def test_synthetic_graphs_are_valid(version: str) -> None:
    for seed in (1, 2, 3):
        data = build_c1_graph(make_deck_spec(seed, 8), version)
        report = check_c1_graph(data)
        assert report.ok, report.findings
        assert report.version == version


def test_synthetic_v1_uses_slug_ids_and_v0_casefolded_ids() -> None:
    spec = make_deck_spec(1, 8)
    v1_ids = [n["id"] for n in build_c1_graph(spec, "v1")["graph"]["nodes"]]
    v0_ids = [n["id"] for n in build_c1_graph(spec, "v0")["graph"]["nodes"]]
    assert v1_ids == [c.slug for c in spec.concepts]
    assert v0_ids == [c.concept_id for c in spec.concepts]
    assert "C1-018" not in rules_of(check_c1_graph(build_c1_graph(spec, "v1")))


Mutation = Callable[[dict], None]


def _node(d: dict, i: int = 0) -> dict:
    return d["graph"]["nodes"][i]


def _edge(d: dict, i: int = 0) -> dict:
    return d["graph"]["edges"][i]


def _prereq_edge(d: dict) -> dict:
    return next(e for e in d["graph"]["edges"] if e["relation_type"] == "prerequisite")


def _set(target: Callable[[dict], dict], key: str, value: object) -> Mutation:
    def mutate(d: dict) -> None:
        target(d)[key] = value

    return mutate


def _drop(target: Callable[[dict], dict], key: str) -> Mutation:
    def mutate(d: dict) -> None:
        del target(d)[key]

    return mutate


def _dup_node(d: dict) -> None:
    d["graph"]["nodes"].append(copy.deepcopy(_node(d)))
    d["graph"]["node_count"] += 1


def _dup_edge(d: dict) -> None:
    d["graph"]["edges"].append(copy.deepcopy(_edge(d)))
    d["graph"]["edge_count"] += 1


def _evidence(value: object) -> Mutation:
    return _set(_prereq_edge, "evidence", value)


# (rule, mutation applied to a valid v1 graph)
BROKEN: list[tuple[str, Mutation]] = [
    ("C1-003", lambda d: d["graph"]["nodes"].append("oops")),
    ("C1-003", _drop(_node, "id")),
    ("C1-003", _set(_node, "label", 5)),
    ("C1-004", _dup_node),
    ("C1-005", _set(lambda d: d["graph"], "node_count", 99)),
    ("C1-005", _set(lambda d: d["graph"], "edge_count", 0)),
    ("C1-006", lambda d: d["graph"]["edges"].append(7)),
    ("C1-006", _drop(_edge, "source")),
    ("C1-006", _set(_edge, "target", None)),
    ("C1-007", _set(_edge, "source", "ghost")),
    ("C1-007", _set(_edge, "target", "ghost")),
    ("C1-008", lambda d: _edge(d).update(target=_edge(d)["source"])),
    ("C1-009", _dup_edge),
    ("C1-010", _set(_edge, "relation_type", "similar")),
    ("C1-010", _set(_edge, "relation_type", ["prerequisite"])),
    ("C1-011", _set(_prereq_edge, "direction", "target_is_prerequisite_of_source")),
    ("C1-012", _set(_prereq_edge, "confidence", 1.5)),
    ("C1-012", _set(_prereq_edge, "confidence", -0.1)),
    ("C1-012", _set(_prereq_edge, "confidence", "high")),
    ("C1-012", _set(_prereq_edge, "confidence", True)),
    ("C1-013", _drop(_prereq_edge, "confidence")),
    ("C1-013", _set(_prereq_edge, "confidence", None)),
    ("C1-014", _set(_prereq_edge, "source_type", "guess")),
    ("C1-015", _drop(_prereq_edge, "source_type")),
    ("C1-016", _evidence("text")),
    ("C1-016", _evidence(["text"])),
    ("C1-016", _evidence([{"page": "1", "sentence": "s"}])),
    ("C1-016", _evidence([{"page": True, "sentence": "s"}])),
    ("C1-016", _evidence([{"page": 1, "sentence": 2}])),
    ("C1-018", _set(_node, "id", "Gradient Descent")),
    ("C1-018", _set(_node, "id", "-bad-")),
    ("C1-019", _drop(_node, "description")),
    ("C1-019", _set(_node, "description", "   ")),
]


@pytest.mark.parametrize(("rule", "mutate"), BROKEN)
def test_each_rule_fires(rule: str, mutate: Mutation, v1: dict) -> None:
    # Slug and description rules need ids that stay consistent, so mutate in place.
    mutate(v1)
    report = check_c1_graph(v1, expected="v1")
    assert rule in rules_of(report)
    severity = next(s for r, s, _ in RULES if r == rule)
    assert any(f.rule == rule and f.severity == severity for f in report.findings)
    assert report.ok == (severity == "warning" and report.stats["errors"] == 0)


def test_optional_prerequisite_fields_do_not_fire_when_valid(v1: dict) -> None:
    for e in v1["graph"]["edges"]:
        e.pop("evidence", None)
        e.pop("direction", None)
    report = check_c1_graph(v1)
    assert report.ok
    assert not report.findings


def test_non_dict_node_under_v1_has_no_description_warning_crash(v1: dict) -> None:
    v1["graph"]["nodes"].append(None)
    report = check_c1_graph(v1)
    assert "C1-003" in rules_of(report)


def test_non_object_payload() -> None:
    report = check_c1_graph([1])
    assert (report.version, report.ok) == ("unknown", False)
    assert rules_of(report) == {"C1-001"}


@pytest.mark.parametrize(
    "payload",
    [{}, {"graph": []}, {"graph": {"nodes": []}}, {"graph": {"nodes": {}, "edges": []}}],
)
def test_missing_graph_parts(payload: object) -> None:
    report = check_c1_graph(payload)
    assert report.version == "unknown"
    assert rules_of(report) == {"C1-002"}


def test_empty_graph_is_v0_and_ok() -> None:
    report = check_c1_graph({"graph": {"nodes": [], "edges": []}})
    assert (report.version, report.ok) == ("v0", True)


def test_v0_expectations(v0: dict) -> None:
    _node(v0)["id"] = "Gradient Descent"
    _edge(v0)["source"] = "Gradient Descent"
    report = check_c1_graph(v0)
    assert report.version == "v0"
    assert rules_of(report) == {"C1-020"}
    assert report.ok


def test_expected_v1_on_v0_data_applies_v1_expectations(v0: dict) -> None:
    report = check_c1_graph(v0, expected="v1")
    assert report.version == "v0"
    assert {"C1-018", "C1-019"} <= rules_of(report)
    assert not report.ok  # spaces in ids are slug errors


def test_expected_v0_on_v1_data(v1: dict) -> None:
    _node(v1)["id"] = "Loss"
    report = check_c1_graph(v1, expected="v0")
    assert "C1-020" in rules_of(report)
    assert "C1-018" not in rules_of(report)


def test_version_detection_descriptions_only_is_v1(v0: dict) -> None:
    _node(v0)["description"] = "A description."
    assert check_c1_graph(v0).version == "v1"


def test_mixed_prerequisites_without_descriptions(v1: dict) -> None:
    for n in v1["graph"]["nodes"]:
        del n["description"]
    report = check_c1_graph(v1)
    assert report.version == "mixed"
    assert "C1-021" in rules_of(report)
    assert not ({"C1-018", "C1-019", "C1-020"} & rules_of(report))
    assert report.ok


def test_mixed_v0_shape_with_slug_ids(v0: dict) -> None:
    _node(v0)["id"] = "gradient-descent"
    _edge(v0)["source"] = "gradient-descent"
    report = check_c1_graph(v0)
    assert report.version == "mixed"
    assert "C1-021" in rules_of(report)


def test_cycle_warning_names_one_cycle(v1: dict) -> None:
    v1["graph"]["edges"].append(
        {
            "source": "backpropagation",
            "target": "loss-function",
            "relation_type": "prerequisite",
            "confidence": 0.5,
            "source_type": "inferred",
        }
    )
    v1["graph"]["edge_count"] += 1
    report = check_c1_graph(v1)
    cycles = [f for f in report.findings if f.rule == "C1-017"]
    assert len(cycles) == 1
    assert cycles[0].severity == "warning"
    assert cycles[0].message == (
        "Prerequisite cycle: backpropagation -> loss-function -> gradient-descent"
        " -> backpropagation."
    )
    assert report.ok


def test_acyclic_graph_with_shared_descendants_has_no_cycle() -> None:
    nodes = [{"id": c, "label": c, "description": "d"} for c in "abcd"]
    pairs = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("a", "d")]
    edges = [
        {
            "source": s,
            "target": t,
            "relation_type": "prerequisite",
            "confidence": 1,
            "source_type": "inferred",
        }
        for s, t in pairs
    ]
    report = check_c1_graph({"graph": {"nodes": nodes, "edges": edges}})
    assert report.ok
    assert not report.findings


def test_self_loop_is_not_reported_as_cycle(v1: dict) -> None:
    e = _prereq_edge(v1)
    e["target"] = e["source"]
    report = check_c1_graph(v1)
    assert "C1-008" in rules_of(report)
    assert "C1-017" not in rules_of(report)


def test_deterministic(v1: dict) -> None:
    _edge(v1)["confidence"] = 9
    assert check_c1_graph(v1) == check_c1_graph(copy.deepcopy(v1))


GARBAGE: list[object] = [
    None,
    [],
    "text",
    42,
    3.5,
    True,
    {"graph": None},
    {"graph": {"nodes": [None, 1, "x", [], {}], "edges": [None, 1, "x", [], {}]}},
    {"graph": {"nodes": [{"id": []}, {"id": {}, "label": {}}], "edges": [{"source": []}]}},
    {"graph": {"nodes": [{"id": "a", "label": "a", "description": []}], "edges": [{"a": 1}]}},
    {
        "graph": {
            "nodes": [{"id": "a", "label": "a"}],
            "edges": [
                {"source": "a", "target": "a", "relation_type": {"x": 1}, "evidence": [[], None]}
            ],
        }
    },
    {"graph": {"node_count": "many", "nodes": [], "edges": [], "edge_count": None}},
]


@pytest.mark.parametrize("payload", GARBAGE)
@pytest.mark.parametrize("expected", ["auto", "v0", "v1"])
def test_never_raises_on_garbage(payload: object, expected: str) -> None:
    report = check_c1_graph(payload, expected=expected)  # type: ignore[arg-type]
    assert isinstance(report, ContractReport)
    assert report.stats["errors"] + report.stats["warnings"] == len(report.findings)
