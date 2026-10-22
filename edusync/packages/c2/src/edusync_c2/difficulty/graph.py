"""Read-only view over a C1 graph: valid prerequisite edges and per-concept metrics.

Only ``relation_type == "prerequisite"`` edges with a correct direction between known nodes
count as prerequisites. ``co_mentioned`` and unknown relation types never do.
"""

from collections.abc import Collection, Mapping

from edusync_c2.difficulty.schemas import C1Edge, C1GraphResponse, C1Node, PrerequisiteMetrics

_PREREQ = "prerequisite"
_DIRECTION = "source_is_prerequisite_of_target"


def _rank(e: C1Edge) -> tuple[int, float]:
    """Preference order for duplicate edges: source type first, then confidence (None lowest)."""
    kind = {"curriculum_defined": 2, "inferred": 1}.get(e.source_type or "", 0)
    return kind, -1.0 if e.confidence is None else e.confidence


class GraphView:
    """Indexes a C1 graph response and answers prerequisite questions.

    ``aliases`` maps an alias concept id to its canonical id and ``dropped`` lists concept ids
    removed from the graph. Duplicate edges are only collapsed when aliases are given.
    Aliases are resolved one hop only: the caller must supply an already-flattened map (no
    chains). An alias whose id or canonical id is not a node is ignored. Edges are validated
    on the raw ids first; surviving prerequisite edges touching a dropped concept are cut (no
    bridging), the rest are remapped to canonical ids, merge self-loops are skipped and
    duplicate (source, target) pairs keep the best edge. Cycles created by merging are kept;
    the metrics are cycle safe.
    """

    def __init__(
        self,
        response: C1GraphResponse,
        *,
        aliases: Mapping[str, str] | None = None,
        dropped: Collection[str] = (),
    ) -> None:
        self.warnings: list[str] = []
        raw: dict[str, C1Node] = {}
        for n in response.graph.nodes:
            raw.setdefault(n.id, n)
        self._raw = raw
        self._canon: dict[str, str] = {
            a: c for a, c in (aliases or {}).items() if a != c and a in raw and c in raw
        }
        gone = frozenset(dropped)
        self._nodes: dict[str, C1Node] = {
            i: n for i, n in raw.items() if i not in self._canon and i not in gone
        }
        self.merged_aliases = len(self._canon)
        self.rewritten_edges = 0
        self.dropped_edges = 0
        self.merge_self_loops = 0
        best: dict[tuple[str, str], C1Edge] = {}
        for e in response.graph.edges:
            if e.relation_type != _PREREQ:
                continue
            problem = self._problem(e)
            if problem:
                if problem not in self.warnings:
                    self.warnings.append(problem)
                continue
            src, tgt = self._canon.get(e.source, e.source), self._canon.get(e.target, e.target)
            if {e.source, e.target, src, tgt} & gone:
                self.dropped_edges += 1
                continue
            if src == tgt:
                self.merge_self_loops += 1
                continue
            if (src, tgt) != (e.source, e.target):
                self.rewritten_edges += 1
                e = e.model_copy(update={"source": src, "target": tgt})
            # Without aliases duplicates are kept as before; dedupe only applies to merges.
            key = (src, tgt) if self._canon else (str(len(best)), "")
            old = best.get(key)
            if old is None or _rank(e) > _rank(old):
                best[key] = e
        if self.merge_self_loops:
            self.warnings.append(
                f"{self.merge_self_loops} prerequisite edge(s) became self-loops after alias "
                "merge and were ignored"
            )
        self._incoming: dict[str, list[C1Edge]] = {}
        self._dependents: dict[str, list[str]] = {}
        for e in best.values():
            self._incoming.setdefault(e.target, []).append(e)
            self._dependents.setdefault(e.source, []).append(e.target)

    def _problem(self, e: C1Edge) -> str | None:
        label = f"'{e.source}' -> '{e.target}'"
        if e.direction not in (None, _DIRECTION):
            return f"ignored prerequisite edge {label}: unsupported direction '{e.direction}'"
        if e.source not in self._raw or e.target not in self._raw:
            return f"ignored prerequisite edge {label}: references an unknown node"
        if e.source == e.target:
            return f"ignored prerequisite edge {label}: self-loop"
        return None

    @property
    def has_prerequisite_edges(self) -> bool:
        return bool(self._incoming)

    def node(self, concept_id: str) -> C1Node | None:
        return self._nodes.get(self._canon.get(concept_id, concept_id))

    def prerequisites_of(self, concept_id: str) -> list[C1Edge]:
        """Direct valid prerequisite edges whose target is the concept, in graph order."""
        return list(self._incoming.get(self._canon.get(concept_id, concept_id), []))

    def dependents_of(self, concept_id: str) -> list[str]:
        """Distinct concepts that directly depend on the concept, in graph order."""
        return list(
            dict.fromkeys(self._dependents.get(self._canon.get(concept_id, concept_id), []))
        )

    def metrics(self, concept_id: str) -> PrerequisiteMetrics | None:
        concept_id = self._canon.get(concept_id, concept_id)
        if concept_id not in self._nodes:
            return None
        direct = self.prerequisites_of(concept_id)
        confs = [e.confidence for e in direct if e.confidence is not None]
        depth, cyclic = self._depth(concept_id, set(), {})
        weighted, _ = self._weighted_depth(concept_id, set(), {})
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
            dependents_count=len(self.dependents_of(concept_id)),
            weighted_depth=weighted,
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

    def _weighted_depth(
        self, cid: str, path: set[str], memo: dict[str, float]
    ) -> tuple[float, bool]:
        """Heaviest prerequisite chain below ``cid`` (sum of edge confidences, None counts 1.0).

        Cycle safe like ``_depth``: an edge back to a node on the current path is ignored, and
        only cycle-free results are cached.
        """
        if cid in memo:
            return memo[cid], False
        path.add(cid)
        best, cyclic = 0.0, False
        for e in self._incoming.get(cid, []):
            if e.source in path:
                cyclic = True
                continue
            w, c = self._weighted_depth(e.source, path, memo)
            weight = 1.0 if e.confidence is None else e.confidence
            best, cyclic = max(best, w + weight), cyclic or c
        path.discard(cid)
        if not cyclic:
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
