from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.pipeline import CONCEPT_CYCLE_WARNING
from edusync_c2.difficulty.schemas import C1GraphResponse
from edusync_c2.difficulty.synthetic import build_c1_graph, build_pptx, make_deck_spec

SPEC = make_deck_spec(3, 6, "lec-cycle")


def _edge(a: str, b: str) -> dict:
    return {
        "source": a,
        "target": b,
        "relation_type": "prerequisite",
        "direction": "source_is_prerequisite_of_target",
        "confidence": 0.8,
    }


def _cyclic() -> C1GraphResponse:
    raw = build_c1_graph(SPEC, "v1")
    ids = [n["id"] for n in raw["graph"]["nodes"]]
    raw["graph"]["edges"] = [_edge(ids[0], ids[1]), _edge(ids[1], ids[2]), _edge(ids[2], ids[0])]
    return C1GraphResponse.model_validate(raw)


def test_cycle_is_reported(caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    resp = DifficultyPipeline().run(build_pptx(SPEC), "deck.pptx", _cyclic())
    flagged = [p for p in resp.predictions if CONCEPT_CYCLE_WARNING in p.warnings]
    assert flagged
    n = len(flagged)
    assert f"{n} concept(s) have a prerequisite cycle in the C1 graph" in resp.warnings
    graph_stage = next(s for s in resp.stages if s.name == "graph")
    assert f"{n} concept(s) with a prerequisite cycle" in graph_stage.detail
    assert any(
        r.levelname == "WARNING" and r.getMessage() == f"graph_cycle concepts={n}"
        for r in caplog.records
    )


def test_acyclic_graph_has_no_cycle_warning():
    graph = C1GraphResponse.model_validate(build_c1_graph(SPEC, "v1"))
    resp = DifficultyPipeline().run(build_pptx(SPEC), "deck.pptx", graph)
    assert not any("cycle" in w for w in resp.warnings)
    assert all(CONCEPT_CYCLE_WARNING not in p.warnings for p in resp.predictions)
