import pytest

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.materials import MaterialsError
from edusync_c2.difficulty.models import RuleBaselineModel
from edusync_c2.difficulty.schemas import (
    AlignmentStatus,
    C1GraphResponse,
    DifficultyLabel,
    DifficultyResponse,
    FeatureStatus,
    ModelKind,
    ModelOutput,
)
from edusync_c2.difficulty.synthetic import build_c1_graph, build_pdf, build_pptx, make_deck_spec

SPEC = make_deck_spec(3, 6, "lec-test")


def graph(version: str) -> C1GraphResponse:
    return C1GraphResponse.model_validate(build_c1_graph(SPEC, version))


def feat(pred, name):
    return next(f for f in pred.features if f.name == name)


def stages(resp):
    return {s.name: s for s in resp.stages}


@pytest.fixture(scope="module")
def pptx() -> bytes:
    return build_pptx(SPEC)


def test_v1_payload(pptx):
    resp = DifficultyPipeline().run(pptx, "deck.pptx", graph("v1"))
    assert resp.lecture_id == "lec-test"
    assert [i.kind for i in resp.inputs] == ["real", "real"]
    assert resp.inputs[0].name == "materials"
    assert len(resp.predictions) == 6
    for p in resp.predictions:
        assert feat(p, "prereq.direct_count").status is FeatureStatus.COMPUTED
        assert p.confidence is None
        assert p.probabilities is None
        assert p.difficulty is not None
        assert p.status == "partial"
        assert p.explanation
    assert list(stages(resp)) == ["materials", "graph", "concepts", "features", "fusion", "model"]
    assert stages(resp)["graph"].status == "ok"
    assert stages(resp)["concepts"].status == "mocked"
    assert stages(resp)["features"].status == "partial"
    assert "not trained" in stages(resp)["model"].detail
    assert resp.todo
    assert not any("mock" in w for w in resp.warnings)


def test_v0_payload_group_b_unavailable(pptx):
    resp = DifficultyPipeline().run(pptx, "deck.pptx", graph("v0"))
    assert any("no prerequisite edges" in w for w in resp.warnings)
    for p in resp.predictions:
        assert feat(p, "prereq.direct_count").status is FeatureStatus.UNAVAILABLE
        assert p.difficulty is not None


def test_mock_provider(pptx):
    resp = DifficultyPipeline().run(pptx, "deck.pptx")
    assert resp.lecture_id is None
    assert resp.inputs[1].kind == "mock"
    assert stages(resp)["graph"].status == "mocked"
    assert any("mock" in w for w in resp.warnings)
    for p in resp.predictions:
        assert feat(p, "prereq.direct_count").status is FeatureStatus.MOCKED
        assert feat(p, "dens.concept_frequency").status is FeatureStatus.MOCKED


def test_pdf_structure_unavailable():
    resp = DifficultyPipeline().run(build_pdf(SPEC), "deck.pdf", graph("v1"))
    assert resp.file_type == "pdf"
    assert stages(resp)["materials"].status == "partial"
    assert resp.warnings
    for p in resp.predictions:
        assert feat(p, "dens.shape_count_mean").status is FeatureStatus.UNAVAILABLE


def test_concept_not_found(pptx):
    payload = build_c1_graph(SPEC, "v1")
    payload["graph"]["nodes"].append({"id": "zzz quark", "label": "Zzz Quark", "frequency": 1})
    resp = DifficultyPipeline().run(pptx, "deck.pptx", C1GraphResponse.model_validate(payload))
    missing = next(p for p in resp.predictions if p.concept_id == "zzz quark")
    assert missing.alignment_status is AlignmentStatus.NOT_FOUND
    assert any("not found" in w for w in missing.warnings)
    assert missing.status in {"failed", "partial"}
    assert "not found in materials" in stages(resp)["concepts"].detail


def test_graph_warnings_surface(pptx):
    payload = build_c1_graph(SPEC, "v1")
    payload["graph"]["edges"].append(
        {"source": "nope", "target": "nada", "relation_type": "prerequisite"}
    )
    resp = DifficultyPipeline().run(pptx, "deck.pptx", C1GraphResponse.model_validate(payload))
    assert any("unknown node" in w for w in resp.warnings)
    assert "unknown node" in stages(resp)["graph"].detail


def test_deterministic_and_roundtrip(pptx):
    pipe = DifficultyPipeline()
    a = pipe.run(pptx, "deck.pptx", graph("v1"))
    b = pipe.run(pptx, "deck.pptx", graph("v1"))
    assert a.model_dump() == b.model_dump()
    assert DifficultyResponse.model_validate_json(a.model_dump_json()) == a


def test_materials_error_propagates():
    with pytest.raises(MaterialsError) as ei:
        DifficultyPipeline().run(b"x", "a.txt")
    assert ei.value.status_code == 400


def test_failed_prediction_when_model_returns_none(pptx):
    class NoneModel:
        info = RuleBaselineModel.info

        def predict(self, inputs):
            return [None for _ in inputs]

    resp = DifficultyPipeline(model=NoneModel()).run(pptx, "deck.pptx", graph("v1"))
    for p in resp.predictions:
        assert p.status == "failed"
        assert p.difficulty is None
        assert p.score is None
        assert p.explanation == []
        assert "not enough computed features for the baseline" in p.warnings
    assert "without a prediction" in stages(resp)["model"].detail


def test_custom_model_and_filter(pptx):
    class Fake:
        info = RuleBaselineModel.info.model_copy(update={"kind": ModelKind.TRAINED, "name": "fake"})

        def predict(self, inputs):
            return [
                ModelOutput(label=DifficultyLabel.HIGH, probabilities={DifficultyLabel.HIGH: 1.0})
                for _ in inputs
            ]

    class DropFirst:
        name = "drop_first"

        def apply(self, concepts):
            return [
                c.model_copy(update={"filter_status": "dropped" if i == 0 else "kept"})
                for i, c in enumerate(concepts)
            ]

    pipe = DifficultyPipeline(model=Fake(), concept_filter=DropFirst(), use_mocked_features=True)
    resp = pipe.run(pptx, "deck.pptx", graph("v1"))
    assert len(resp.predictions) == 5
    assert resp.predictions[0].probabilities == {DifficultyLabel.HIGH: 1.0}
    assert stages(resp)["concepts"].status == "ok"
    assert "not trained" not in stages(resp)["model"].detail
    assert pipe.info().concept_filter == "drop_first"


def test_all_computed_gives_ok_status(pptx, monkeypatch):
    import edusync_c2.difficulty.pipeline as mod

    real = mod.compute_all_features

    def all_computed(*a, **k):
        return [
            f.model_copy(update={"status": FeatureStatus.COMPUTED, "value": f.value or 0.0})
            for f in real(*a, **k)
        ]

    monkeypatch.setattr(mod, "compute_all_features", all_computed)
    resp = DifficultyPipeline().run(pptx, "deck.pptx", graph("v1"))
    assert all(p.status == "ok" for p in resp.predictions)
    assert stages(resp)["features"].status == "ok"


def test_info():
    info = DifficultyPipeline().info()
    assert info.model.name == "rule_baseline"
    assert info.graph_provider == "mock_from_slide_titles"
    assert info.concept_filter == "mock_keep_all"
    assert len(info.features) == 24
    assert info.todo
