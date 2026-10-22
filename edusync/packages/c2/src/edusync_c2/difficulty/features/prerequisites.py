"""Group B: prerequisite features from the C1 graph (prerequisite edges only)."""

from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import (
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    FeatureValue,
    InputSource,
    PrerequisiteMetrics,
)

SOURCE = "c1_graph:prerequisite_edges"


def _unavailable(name: str, note: str) -> FeatureValue:
    return FeatureValue(
        name=name,
        group=FeatureGroup.PREREQUISITES,
        status=FeatureStatus.UNAVAILABLE,
        source=SOURCE,
        note=note,
    )


def prerequisite_features(
    concept: ConceptRecord,
    metrics: PrerequisiteMetrics | None,
    graph_has_prerequisite_edges: bool,
    graph_source: InputSource,
) -> list[FeatureValue]:
    """Group B features in registry order; mock graph -> MOCKED, caller graph -> COMPUTED."""
    names = features_of(FeatureGroup.PREREQUISITES)
    if not graph_has_prerequisite_edges:
        note = "C1 graph has no prerequisite edges (waiting for C1)"
        return [_unavailable(n, note) for n in names]
    if metrics is None:
        return [
            _unavailable(n, f"concept id '{concept.id}' is unknown to the graph") for n in names
        ]
    mocked = graph_source.kind == "mock"
    status = FeatureStatus.MOCKED if mocked else FeatureStatus.COMPUTED
    note = "computed from a mock graph" if mocked else ""
    src = "mock:c1_graph" if mocked else SOURCE
    values: list[float | None] = [
        float(metrics.direct_count),
        float(metrics.chain_depth),
        float(metrics.transitive_count),
        metrics.mean_confidence,
        metrics.curriculum_defined_share,
    ]
    out: list[FeatureValue] = []
    for name, value in zip(names, values, strict=True):
        if value is None or (metrics.direct_count == 0 and name.endswith(("confidence", "share"))):
            out.append(_unavailable(name, "no direct prerequisites"))
        else:
            out.append(
                FeatureValue(
                    name=name,
                    group=FeatureGroup.PREREQUISITES,
                    value=value,
                    status=status,
                    source=src,
                    note=note,
                )
            )
    return out
