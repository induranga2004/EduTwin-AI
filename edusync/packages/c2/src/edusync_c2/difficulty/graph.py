"""Read-only view over a C1 graph: valid prerequisite edges and per-concept metrics.

Only ``relation_type == "prerequisite"`` edges with a correct direction between known nodes
count as prerequisites. ``co_mentioned`` and unknown relation types never do.
"""

from edusync_c2.difficulty.schemas import C1Edge, C1GraphResponse, C1Node, PrerequisiteMetrics

_PREREQ = "prerequisite"
_DIRECTION = "source_is_prerequisite_of_target"


class GraphView:
    """Indexes a C1 graph response and answers prerequisite questions."""

    def __init__(self, response: C1GraphResponse) -> None:
        self.warnings: list[str] = []
        self._nodes: dict[str, C1Node] = {}
        for n in response.graph.nodes:
            self._nodes.setdefault(n.id, n)
        self._incoming: dict[str, list[C1Edge]] = {}
        for e in response.graph.edges:
            if e.relation_type != _PREREQ:
                continue
            problem = self._problem(e)
            if problem:
                if problem not in self.warnings:
                    self.warnings.append(problem)
                continue
            self._incoming.setdefault(e.target, []).append(e)

    def _problem(self, e: C1Edge) -> str | None:
        label = f"'{e.source}' -> '{e.target}'"
        if e.direction not in (None, _DIRECTION):
            return f"ignored prerequisite edge {label}: unsupported direction '{e.direction}'"
        if e.source not in self._nodes or e.target not in self._nodes:
            return f"ignored prerequisite edge {label}: references an unknown node"
        if e.source == e.target:
            return f"ignored prerequisite edge {label}: self-loop"
        return None

    @property
    def has_prerequisite_edges(self) -> bool:
        return bool(self._incoming)

    def node(self, concept_id: str) -> C1Node | None:
        return self._nodes.get(concept_id)

    def prerequisites_of(self, concept_id: str) -> list[C1Edge]:
        """Direct valid prerequisite edges whose target is the concept, in graph order."""
        return list(self._incoming.get(concept_id, []))

    def metrics(self, concept_id: str) -> PrerequisiteMetrics | None:
        if concept_id not in self._nodes:
            return None
        direct = self.prerequisites_of(concept_id)
        confs = [e.confidence for e in direct if e.confidence is not None]
        depth, cyclic = self._depth(concept_id, set(), {})
        return PrerequisiteMetrics(
            direct_count=len(direct),
            chain_depth=depth,
            transitive_count=len(self._reachable(concept_id)),
            mean_confidence=sum(confs) / len(confs) if confs else None,
            curriculum_defined_share=(
                sum(e.source_type == "curriculum_defined" for e in direct) / len(direct)
                if direct
                else None
            ),
            has_cycle=cyclic,
        )

    def _depth(self, cid: str, path: set[str], memo: dict[str, int]) -> tuple[int, bool]:
        """Longest prerequisite chain below ``cid`` and whether a cycle was met."""
        if cid in memo:
            return memo[cid], False
        path.add(cid)
        best, cyclic = 0, False
        for e in self._incoming.get(cid, []):
            if e.source in path:
                cyclic = True
                continue
            d, c = self._depth(e.source, path, memo)
            best, cyclic = max(best, d + 1), cyclic or c
        path.discard(cid)
        if not cyclic:  # results below a cycle depend on the path, so only cache clean ones
            memo[cid] = best
        return best, cyclic

    def _reachable(self, cid: str) -> set[str]:
        seen: set[str] = set()
        stack = [cid]
        while stack:
            for e in self._incoming.get(stack.pop(), []):
                if e.source not in seen:
                    seen.add(e.source)
                    stack.append(e.source)
        seen.discard(cid)
        return seen
