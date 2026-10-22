"""Find diagrams on a slide: boxes joined by arrows, with text labels, written as sentences.

Geometry only, no machine learning. Rules (all limits are documented constants):
  * Nodes are boxes (geo shapes, sticky notes, frames) and free-standing text.
  * A text shape whose centre lies inside a box is that box's label.
  * An arrow end is attached to a shape when BBB says so (a "binding"), or else to the nearest
    node within `SNAP_DISTANCE` of the end point.
  * Direction comes from the arrowheads: head at the end means start -> end, head only at the
    start means end -> start, both means a two-way link, none means a plain line.
  * A diagram is a set of nodes connected by arrows; a lone box is not a diagram.
  * Plain one-way arrows that follow each other are written as one chain
    (A -> B -> C). A branch, a labelled arrow, a two-way link or a plain line keeps its
    own sentence.

Known limits: curved arrows and lines (not arrows) are not followed; rotated shapes use their
unrotated box; a very large box (more than `MAX_CONTAINER_AREA`) is not treated as a container so
a big frame cannot swallow a whole slide; text that is not inside any box and not at an arrow is
not part of a diagram.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from edusync_c4.schemas import (
    BoundingBox,
    Diagram,
    DiagramEdge,
    DiagramNode,
    NormalizedAnnotation,
    ShapeType,
)

SNAP_DISTANCE = 40.0
LABEL_PAD = 10.0
MAX_CONTAINER_AREA = 400_000.0
NODE_TYPES = {ShapeType.GEO, ShapeType.NOTE, ShapeType.FRAME}
KIND_NAMES = {ShapeType.GEO: "box", ShapeType.NOTE: "note", ShapeType.FRAME: "frame"}


def _props(ann: NormalizedAnnotation) -> dict[str, Any]:
    props = ann.shape_state.get("props")
    return props if isinstance(props, dict) else {}


def _area(b: BoundingBox) -> float:
    return b.width * b.height


def _center_inside(outer: BoundingBox, inner: BoundingBox, pad: float = LABEL_PAD) -> bool:
    cx, cy = inner.x + inner.width / 2, inner.y + inner.height / 2
    return (
        outer.x - pad <= cx <= outer.x + outer.width + pad
        and outer.y - pad <= cy <= outer.y + outer.height + pad
    )


def _distance_to_box(px: float, py: float, b: BoundingBox) -> float:
    dx = max(b.x - px, 0.0, px - (b.x + b.width))
    dy = max(b.y - py, 0.0, py - (b.y + b.height))
    return math.hypot(dx, dy)


@dataclass
class _Structure:
    nodes: dict[str, NormalizedAnnotation] = field(default_factory=dict)
    label_owner: dict[str, str] = field(default_factory=dict)  # label shape id -> node id
    labels: dict[str, list[NormalizedAnnotation]] = field(default_factory=dict)
    arrows: list[NormalizedAnnotation] = field(default_factory=list)
    # arrow id -> (start node id | None, end node id | None)
    ends: dict[str, tuple[str | None, str | None]] = field(default_factory=dict)


def _latest(annotations: list[NormalizedAnnotation]) -> list[NormalizedAnnotation]:
    """One annotation per shape: the newest non-deleted version."""
    best: dict[str, NormalizedAnnotation] = {}
    for a in annotations:
        if a.is_deleted or a.is_clear:
            continue
        old = best.get(a.shape_id)
        if old is None or a.timestamp_seconds >= old.timestamp_seconds:
            best[a.shape_id] = a
    return list(best.values())


def _resolve_end(point: Any, arrow: NormalizedAnnotation, s: _Structure) -> str | None:
    if isinstance(point, dict) and point.get("type") == "binding":
        bound = str(point.get("boundShapeId", ""))
        bound = s.label_owner.get(bound, bound)
        return bound if bound in s.nodes else None
    if not isinstance(point, dict):
        return None
    ox = float(arrow.shape_state.get("x", 0.0) or 0.0)
    oy = float(arrow.shape_state.get("y", 0.0) or 0.0)
    px, py = ox + float(point.get("x", 0.0)), oy + float(point.get("y", 0.0))
    best: tuple[float, float, str] | None = None
    for node_id, node in s.nodes.items():
        d = _distance_to_box(px, py, node.bounds)
        if d <= SNAP_DISTANCE:
            key = (d, _area(node.bounds), node_id)
            if best is None or key < best:
                best = key
    return best[2] if best else None


def _analyse(annotations: list[NormalizedAnnotation]) -> _Structure:
    shapes = _latest(annotations)
    s = _Structure()
    boxes = {
        a.shape_id: a
        for a in shapes
        if a.shape_type in NODE_TYPES and 0 < _area(a.bounds) <= MAX_CONTAINER_AREA
    }
    texts = [a for a in shapes if a.shape_type == ShapeType.TEXT]
    for t in texts:
        owners = [b for b in boxes.values() if _center_inside(b.bounds, t.bounds)]
        if owners:
            owner = min(owners, key=lambda b: (_area(b.bounds), b.shape_id))
            s.label_owner[t.shape_id] = owner.shape_id
            s.labels.setdefault(owner.shape_id, []).append(t)
    s.nodes = dict(boxes)
    for t in texts:
        if t.shape_id not in s.label_owner:
            s.nodes[t.shape_id] = t  # free-standing text can be an end of an arrow
    s.arrows = sorted(
        (a for a in shapes if a.shape_type == ShapeType.ARROW),
        key=lambda a: (a.timestamp_seconds, a.shape_id),
    )
    for arrow in s.arrows:
        props = _props(arrow)
        s.ends[arrow.shape_id] = (
            _resolve_end(props.get("start"), arrow, s),
            _resolve_end(props.get("end"), arrow, s),
        )
    return s


def _node_name(node: NormalizedAnnotation, s: _Structure) -> str:
    parts: list[str] = []
    own = _props(node).get("text")
    if isinstance(own, str) and own.strip():
        parts.append(own.strip())
    for label in sorted(s.labels.get(node.shape_id, []), key=lambda t: (t.bounds.y, t.bounds.x)):
        if label.text_content and label.text_content.strip():
            parts.append(label.text_content.strip())
    own_text = node.text_content.strip() if node.text_content else ""
    if node.shape_type == ShapeType.TEXT and own_text and own_text not in parts:
        parts.append(own_text)
    if node.shape_type == ShapeType.NOTE and not parts and node.text_content:
        parts.append(node.text_content.strip())
    return (
        " ".join(" ".join(parts).split())
        or f"(unlabelled {KIND_NAMES.get(node.shape_type, 'item')})"
    )


def structural_links(annotations: list[NormalizedAnnotation]) -> list[tuple[str, str]]:
    """Pairs of shape ids that belong together by structure: a label and its box, an arrow and
    each shape it is attached to. Used by grouping, which ignores drawing time for these."""
    s = _analyse(annotations)
    links: list[tuple[str, str]] = [(label, owner) for label, owner in s.label_owner.items()]
    for arrow_id, (start, end) in s.ends.items():
        for node_id in (start, end):
            if node_id:
                links.append((arrow_id, node_id))
    return links


def _chain_sentences(edges: list[DiagramEdge]) -> list[str]:
    """One sentence per chain of plain one-way arrows, one per other edge (in drawing order)."""

    def plain(e: DiagramEdge) -> bool:
        return e.relation == "directed" and not e.label

    used: set[int] = set()
    sentences: list[str] = []
    has_incoming = {e.target_id for e in edges if plain(e)}

    def extend(first: int) -> None:
        used.add(first)
        names = [edges[first].source_name, edges[first].target_name]
        node = edges[first].target_id
        while True:
            nxt = next(
                (
                    i
                    for i, e in enumerate(edges)
                    if i not in used and plain(e) and e.source_id == node
                ),
                None,
            )
            if nxt is None:
                break
            used.add(nxt)
            names.append(edges[nxt].target_name)
            node = edges[nxt].target_id
        sentences.append(" \u2192 ".join(names))

    def other(i: int) -> None:
        e = edges[i]
        used.add(i)
        arrow = {"directed": "\u2192", "both": "\u2194", "undirected": "\u2014"}[e.relation]
        text = f"{e.source_name} {arrow} {e.target_name}"
        sentences.append(text + (f" ({e.label})" if e.label else ""))

    # chains start where nothing arrives; what is left (branches, loops) starts its own line
    for starts_only in (True, False):
        for i, e in enumerate(edges):
            if i in used:
                continue
            if not plain(e):
                other(i)
            elif not starts_only or e.source_id not in has_incoming:
                extend(i)
    return sentences


def build_diagrams(annotations: list[NormalizedAnnotation]) -> list[Diagram]:
    """Diagrams among the given annotations (use the shapes visible at one moment)."""
    s = _analyse(annotations)
    edges: list[DiagramEdge] = []
    dangling: list[str] = []
    for arrow in s.arrows:
        start, end = s.ends[arrow.shape_id]
        if not start or not end or start == end:
            dangling.append(arrow.shape_id)
            continue
        props = _props(arrow)
        head_end = str(props.get("arrowheadEnd", "arrow")) != "none"
        head_start = str(props.get("arrowheadStart", "none")) != "none"
        if head_start and not head_end:
            source, target, relation = end, start, "directed"
        elif head_start and head_end:
            source, target, relation = start, end, "both"
        elif head_end:
            source, target, relation = start, end, "directed"
        else:
            source, target, relation = start, end, "undirected"
        label = props.get("text")
        edges.append(
            DiagramEdge(
                arrow_shape_id=arrow.shape_id,
                source_id=source,
                target_id=target,
                source_name=_node_name(s.nodes[source], s),
                target_name=_node_name(s.nodes[target], s),
                relation=relation,
                label=label.strip() if isinstance(label, str) and label.strip() else None,
            )
        )

    # connected components over the edges
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in edges:
        parent[find(e.source_id)] = find(e.target_id)
    components: dict[str, list[DiagramEdge]] = {}
    for e in edges:
        components.setdefault(find(e.source_id), []).append(e)

    by_id = {a.shape_id: a for a in _latest(annotations)}
    arrow_ids = {a.shape_id for a in s.arrows}
    diagrams: list[Diagram] = []
    for comp_edges in components.values():
        node_ids = sorted({n for e in comp_edges for n in (e.source_id, e.target_id)})
        member_ids = set(node_ids) | {e.arrow_shape_id for e in comp_edges}
        for node_id in node_ids:
            member_ids |= {t.shape_id for t in s.labels.get(node_id, [])}
        times = [by_id[m].timestamp_seconds for m in member_ids if m in by_id]
        comp_edges.sort(key=lambda e: (by_id[e.arrow_shape_id].timestamp_seconds, e.arrow_shape_id))
        sentences = _chain_sentences(comp_edges)
        digest = hashlib.sha1("|".join(node_ids).encode("utf-8")).hexdigest()[:10]
        diagrams.append(
            Diagram(
                diagram_id=f"dg:{digest}",
                nodes=[
                    DiagramNode(
                        shape_id=n,
                        name=_node_name(s.nodes[n], s),
                        shape_type=s.nodes[n].shape_type,
                        bounds=s.nodes[n].bounds,
                    )
                    for n in node_ids
                ],
                edges=comp_edges,
                sentences=sentences,
                shape_ids=sorted(member_ids),
                dangling_arrow_ids=sorted(
                    a for a in dangling if a in arrow_ids and _touches(a, node_ids, s)
                ),
                first_seconds=min(times) if times else 0.0,
            )
        )
    diagrams.sort(key=lambda d: (d.first_seconds, d.diagram_id))
    return diagrams


def _touches(arrow_id: str, node_ids: list[str], s: _Structure) -> bool:
    start, end = s.ends.get(arrow_id, (None, None))
    return (start in node_ids) or (end in node_ids)
