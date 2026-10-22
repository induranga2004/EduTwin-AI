"""Evidence confidence in [0, 1]: how much evidence backs a score. Not a probability of being right.

confidence = graph_factor * (0.40 * available_share
                             + 0.30 * (1 - min(1, spread))
                             + 0.15 * (1 - min(1, |T1 - T2 percentile| / 0.5))
                             + 0.15 * reference_factor)

* available_share: available signals out of the 5 (A, B, C, D, text);
* spread: (max - min of the available signal scores) / 0.8, so signals that agree add confidence;
* the T1/T2 term is 0.5 (neutral) when fewer than two text models answered;
* reference_factor: 0.4 for a provisional reference (fewer than 30 decks), else 1.0;
* graph_factor: 0.9 for a hand-made (curated) graph, 0.6 for the slide-title mock, else 1.0.
"""

from collections.abc import Mapping

SIGNAL_COUNT = 5
SPREAD_SCALE = 0.8
GAP_SCALE = 0.5
GRAPH_FACTOR = {"curated": 0.9, "mock": 0.6}
PROVISIONAL_REFERENCE_FACTOR = 0.4
WEIGHTS = {"available": 0.40, "agreement": 0.30, "models": 0.15, "reference": 0.15}


def evidence_confidence(
    signals: Mapping[str, float | None],
    t1: float | None,
    t2: float | None,
    *,
    provisional: bool,
    graph_kind: str,
) -> tuple[float, dict[str, float | None]]:
    """``(confidence, parts)``; ``parts`` holds the inputs and each weighted term for the UI."""
    scores = [v for v in signals.values() if v is not None]
    share = len(scores) / SIGNAL_COUNT
    spread = (max(scores) - min(scores)) / SPREAD_SCALE if len(scores) > 1 else 0.0
    gap = None if t1 is None or t2 is None else abs(t1 - t2)
    model_term = 0.5 if gap is None else 1.0 - min(1.0, gap / GAP_SCALE)
    reference = PROVISIONAL_REFERENCE_FACTOR if provisional else 1.0
    graph = GRAPH_FACTOR.get(graph_kind, 1.0)
    terms = {
        "available": WEIGHTS["available"] * share,
        "agreement": WEIGHTS["agreement"] * (1.0 - min(1.0, spread)),
        "models": WEIGHTS["models"] * model_term,
        "reference": WEIGHTS["reference"] * reference,
    }
    value = graph * sum(terms.values())
    parts = {
        "available_share": share,
        "spread": min(1.0, spread),
        "t1_t2_gap": gap,
        "reference_factor": reference,
        "graph_factor": graph,
        **{f"term_{k}": v for k, v in terms.items()},
    }
    return max(0.0, min(1.0, value)), parts
