"""Check a C1 graph payload against the C1 -> C2 graph contract (v0 and v1).

Pure functions on plain JSON-like data: no I/O, deterministic, never raises on bad input.
The ``RULES`` table is the single source for rule ids, severities and descriptions; the contract
document copies it.
"""

import re
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["error", "warning"]
Version = Literal["v0", "v1", "mixed", "unknown"]

RULES: list[tuple[str, Severity, str]] = [
    ("C1-001", "error", "The payload is a JSON object."),
    ("C1-002", "error", "The payload has a `graph` object with `nodes` and `edges` lists."),
    ("C1-003", "error", "Every node is an object with a string `id` and a string `label`."),
    ("C1-004", "error", "Node ids are unique."),
    ("C1-005", "warning", "`node_count` and `edge_count` match the list lengths."),
    ("C1-006", "error", "Every edge is an object with a string `source` and a string `target`."),
    ("C1-007", "error", "Every edge `source` and `target` is the id of a node."),
    ("C1-008", "error", "No edge points from a node to itself."),
    ("C1-009", "error", "No two edges share the same source, target and relation_type."),
    ("C1-010", "error", "`relation_type` is `prerequisite` or `co_mentioned`."),
    (
        "C1-011",
        "error",
        "A prerequisite edge `direction`, when present, is `source_is_prerequisite_of_target`.",
    ),
    (
        "C1-012",
        "error",
        "A prerequisite edge `confidence`, when present, is a number from 0 to 1.",
    ),
    ("C1-013", "warning", "A prerequisite edge has a `confidence`."),
    (
        "C1-014",
        "error",
        "A prerequisite edge `source_type`, when present, is `curriculum_defined` or `inferred`.",
    ),
    ("C1-015", "warning", "A prerequisite edge has a `source_type`."),
    (
        "C1-016",
        "error",
        "Edge `evidence`, when present, is a list of objects with an integer `page` and a string"
        " `sentence`.",
    ),
    ("C1-017", "warning", "The prerequisite edges contain no cycle."),
    ("C1-018", "error", "v1: node ids are slugs (lowercase letters, digits and hyphens)."),
    ("C1-019", "warning", "v1: every node has a non-empty `description`."),
    ("C1-020", "warning", "v0: node ids are casefolded."),
    (
        "C1-021",
        "warning",
        "The graph does not mix v0 and v1 shapes (prerequisite edges without node descriptions,"
        " or slug ids with no v1 field).",
    ),
]

_SEVERITY = {rule: severity for rule, severity, _ in RULES}
_RELATIONS = ("prerequisite", "co_mentioned")
_DIRECTION = "source_is_prerequisite_of_target"
_SOURCE_TYPES = ("curriculum_defined", "inferred")
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

_Add = Callable[[str, str, str], None]


class Finding(BaseModel):
    rule: str
    severity: Literal["error", "warning"]
    message: str
    location: str = ""


class ContractReport(BaseModel):
    version: Version
    ok: bool
    findings: list[Finding] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _report(
    version: Version, findings: list[Finding], counts: tuple[int, int, int, int] = (0, 0, 0, 0)
) -> ContractReport:
    errors = sum(f.severity == "error" for f in findings)
    stats = {
        "nodes": counts[0],
        "edges": counts[1],
        "prerequisite_edges": counts[2],
        "co_mentioned_edges": counts[3],
        "errors": errors,
        "warnings": len(findings) - errors,
    }
    return ContractReport(version=version, ok=errors == 0, findings=findings, stats=stats)


def _find_cycle(pairs: list[tuple[str, str]]) -> list[str] | None:
    """Return one prerequisite cycle as a closed path of ids, or None. Deterministic."""
    adjacency: dict[str, list[str]] = {}
    for source, target in pairs:
        adjacency.setdefault(source, []).append(target)
    state: dict[str, int] = {}  # 1 = on the current path, 2 = finished
    for start in sorted(adjacency):
        if start in state:
            continue
        state[start] = 1
        path = [start]
        iters = [iter(sorted(adjacency.get(start, [])))]
        while path:
            nxt = next(iters[-1], None)
            if nxt is None:
                state[path.pop()] = 2
                iters.pop()
            elif state.get(nxt) == 1:
                return [*path[path.index(nxt) :], nxt]
            elif nxt not in state:
                state[nxt] = 1
                path.append(nxt)
                iters.append(iter(sorted(adjacency.get(nxt, []))))
    return None


def _check_prerequisite(edge: dict, where: str, add: _Add) -> None:
    if "direction" in edge and edge["direction"] != _DIRECTION:
        add("C1-011", f"direction is {edge['direction']!r}, expected {_DIRECTION!r}.", where)
    if edge.get("confidence") is None:
        add("C1-013", "Prerequisite edge has no confidence.", where)
    else:
        value = edge["confidence"]
        if not _is_number(value) or not 0 <= value <= 1:
            add("C1-012", f"confidence {value!r} is not a number in [0, 1].", where)
    if edge.get("source_type") is None:
        add("C1-015", "Prerequisite edge has no source_type.", where)
    elif edge["source_type"] not in _SOURCE_TYPES:
        add("C1-014", f"source_type {edge['source_type']!r} is not one of {_SOURCE_TYPES}.", where)


def _check_evidence(edge: dict, where: str, add: _Add) -> None:
    evidence = edge.get("evidence")
    if evidence is None:
        return
    if not isinstance(evidence, list):
        add("C1-016", "evidence is not a list.", where)
        return
    for k, item in enumerate(evidence):
        ok = (
            isinstance(item, dict)
            and isinstance(item.get("page"), int)
            and not isinstance(item["page"], bool)
            and isinstance(item.get("sentence"), str)
        )
        if not ok:
            add(
                "C1-016",
                "evidence item needs an integer page and a string sentence.",
                f"{where}.evidence[{k}]",
            )


def check_c1_graph(
    payload: object, expected: Literal["auto", "v0", "v1"] = "auto"
) -> ContractReport:
    """Validate a parsed C1 graph response. Always returns a report, never raises."""
    findings: list[Finding] = []

    def add(rule: str, message: str, location: str = "") -> None:
        findings.append(
            Finding(rule=rule, severity=_SEVERITY[rule], message=message, location=location)
        )

    if not isinstance(payload, dict):
        add("C1-001", f"Payload is {type(payload).__name__}, expected a JSON object.")
        return _report("unknown", findings)
    graph = payload.get("graph")
    if not isinstance(graph, dict):
        add("C1-002", "`graph` is missing or not an object.", "graph")
        return _report("unknown", findings)
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        add("C1-002", "`graph.nodes` and `graph.edges` must both be lists.", "graph")
        return _report("unknown", findings)

    ids: list[str] = []
    seen: set[str] = set()
    has_description = False
    slugged = False  # some id is the hyphenated slug of its label
    for i, node in enumerate(nodes):
        where = f"graph.nodes[{i}]"
        if not isinstance(node, dict):
            add("C1-003", "Node is not an object.", where)
            continue
        has_description = has_description or node.get("description") is not None
        node_id = node.get("id")
        label = node.get("label")
        if isinstance(node_id, str) and isinstance(label, str):
            folded = label.casefold()
            slugged = slugged or (node_id != folded and node_id == "-".join(folded.split()))
        if not isinstance(node_id, str) or not isinstance(node.get("label"), str):
            add("C1-003", "Node needs a string id and a string label.", where)
        if isinstance(node_id, str):
            if node_id in seen:
                add("C1-004", f"Duplicate node id {node_id!r}.", where)
            seen.add(node_id)
            ids.append(node_id)

    for key, size in (("node_count", len(nodes)), ("edge_count", len(edges))):
        if key in graph and graph[key] != size:
            add(
                "C1-005",
                f"{key} is {graph[key]!r} but the list has {size} entries.",
                f"graph.{key}",
            )

    pairs: list[tuple[str, str]] = []
    keys: set[tuple[str, str, str]] = set()
    prereq = co = 0
    for i, edge in enumerate(edges):
        where = f"graph.edges[{i}]"
        if not isinstance(edge, dict):
            add("C1-006", "Edge is not an object.", where)
            continue
        source, target, relation = edge.get("source"), edge.get("target"), edge.get("relation_type")
        relation_ok = isinstance(relation, str) and relation in _RELATIONS
        if relation == "prerequisite":
            prereq += 1
        elif relation == "co_mentioned":
            co += 1
        if "relation_type" in edge and not relation_ok:
            add("C1-010", f"relation_type {relation!r} is not one of {_RELATIONS}.", where)
        if not isinstance(source, str) or not isinstance(target, str):
            add("C1-006", "Edge needs a string source and a string target.", where)
        else:
            for role, endpoint in (("source", source), ("target", target)):
                if endpoint not in seen:
                    add("C1-007", f"Edge {role} {endpoint!r} is not a node id.", where)
            if source == target:
                add("C1-008", f"Edge from {source!r} to itself.", where)
            if relation_ok:
                key = (source, target, relation)
                if key in keys:
                    add("C1-009", f"Duplicate {relation} edge {source!r} -> {target!r}.", where)
                keys.add(key)
                if relation == "prerequisite" and source != target:
                    pairs.append((source, target))
        if relation == "prerequisite":
            _check_prerequisite(edge, where, add)
        _check_evidence(edge, where, add)

    cycle = _find_cycle(pairs)
    if cycle is not None:
        add("C1-017", "Prerequisite cycle: " + " -> ".join(cycle) + ".", "graph.edges")

    v1_signal = prereq > 0 or has_description
    version: Version
    if (prereq > 0 and not has_description) or (slugged and not v1_signal):
        version = "mixed"
        add(
            "C1-021",
            "Graph mixes v0 and v1 shapes (prerequisite edges without descriptions, or"
            " slug ids without any v1 field).",
            "graph",
        )
    else:
        version = "v1" if v1_signal else "v0"

    profile = version if expected == "auto" else expected
    if profile == "v1":
        for node_id in ids:
            if not _SLUG.match(node_id):
                add("C1-018", f"Node id {node_id!r} is not a lowercase slug.", "graph.nodes")
        for i, node in enumerate(nodes):
            description = node.get("description") if isinstance(node, dict) else "-"
            if not (isinstance(description, str) and description.strip()):
                add("C1-019", "Node has no description.", f"graph.nodes[{i}]")
    elif profile == "v0":
        for node_id in ids:
            if node_id != node_id.casefold():
                add("C1-020", f"Node id {node_id!r} is not casefolded.", "graph.nodes")

    return _report(version, findings, (len(nodes), len(edges), prereq, co))
