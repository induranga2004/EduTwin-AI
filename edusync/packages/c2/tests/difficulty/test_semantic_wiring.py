"""Semantic scorer wiring: settings, pipeline, router construction, baseline guard."""

import importlib
import zlib

import numpy as np
import pytest
from pydantic import ValidationError

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.features.semantic import CLASSES, SemanticScorer, load_prototypes
from edusync_c2.difficulty.models import SUPPORT_FEATURES
from edusync_c2.difficulty.pipeline import SEMANTIC_TRUNCATED_WARNING
from edusync_c2.difficulty.registry import FEATURE_SPECS
from edusync_c2.difficulty.schemas import C1GraphResponse, FeatureStatus
from edusync_c2.difficulty.synthetic import build_c1_graph, build_pdf, build_pptx, make_deck_spec
from edusync_c2.router import get_pipeline
from edusync_c2.settings import C2Settings, get_c2_settings

RT = importlib.import_module("edusync_c2.router")
SEMANTIC = (
    "expl.semantic_definition_count",
    "expl.semantic_example_count",
    "expl.semantic_support_score",
)
SPEC = make_deck_spec(3, 6, "lec-test")
DECK = build_pptx(SPEC)
GRAPH = C1GraphResponse.model_validate(build_c1_graph(SPEC, "v1"))


class HashEncoder:
    """Deterministic pseudo-random unit vectors per text (prototypes included)."""

    dims = 16
    sha = "deadbeef"

    def encode(self, texts):
        out = np.empty((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            vec = np.random.default_rng(zlib.crc32(text.encode())).normal(size=self.dims)
            out[row] = vec / np.linalg.norm(vec)
        return out


def hash_scorer(**kw) -> SemanticScorer:
    kw.setdefault("def_threshold", 0.0)
    kw.setdefault("ex_threshold", 0.0)
    return SemanticScorer(HashEncoder(), **kw)


def feat(pred, name):
    return next(f for f in pred.features if f.name == name)


# ------------------------------------------------------------------ settings


def test_settings_defaults(monkeypatch):
    for k in (
        "SEMANTIC",
        "SEMANTIC_MODEL_DIR",
        "SEMANTIC_DEF_THRESHOLD",
        "SEMANTIC_EX_THRESHOLD",
        "SEMANTIC_MAX_SENTENCES",
    ):
        monkeypatch.delenv(f"EDUSYNC_C2_{k}", raising=False)
    s = C2Settings(_env_file=None)
    assert (s.semantic, s.semantic_model_dir) == ("off", None)
    assert (s.semantic_def_threshold, s.semantic_ex_threshold) == (0.23, 0.23)
    assert s.semantic_max_sentences == 400


def test_settings_env(monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC", "minilm")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_MODEL_DIR", "some/dir")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_DEF_THRESHOLD", "0.3")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_EX_THRESHOLD", "0")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_MAX_SENTENCES", "7")
    s = C2Settings(_env_file=None)
    assert s.semantic == "minilm"
    assert str(s.semantic_model_dir).replace("\\", "/") == "some/dir"
    assert (s.semantic_def_threshold, s.semantic_ex_threshold) == (0.3, 0.0)
    assert s.semantic_max_sentences == 7


def test_empty_semantic_dir_is_none(monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_MODEL_DIR", "")
    assert C2Settings(_env_file=None).semantic_model_dir is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SEMANTIC", "gpu"),
        ("SEMANTIC_DEF_THRESHOLD", "1.1"),
        ("SEMANTIC_DEF_THRESHOLD", "-0.1"),
        ("SEMANTIC_EX_THRESHOLD", "2"),
        ("SEMANTIC_MAX_SENTENCES", "0"),
    ],
)
def test_settings_reject_bad_values(monkeypatch, name, value):
    monkeypatch.setenv(f"EDUSYNC_C2_{name}", value)
    with pytest.raises(ValidationError, match=name.lower()):
        C2Settings(_env_file=None)


# ------------------------------------------------------------------ pipeline


def test_default_pipeline_has_no_scorer():
    pipe = DifficultyPipeline()
    assert pipe.semantic is None
    assert pipe.semantic_status == "off"
    resp = pipe.run(DECK, "deck.pptx", GRAPH)
    for p in resp.predictions:
        for name in SEMANTIC:
            f = feat(p, name)
            assert f.status is FeatureStatus.UNAVAILABLE
            assert f.value is None
            assert f.source == "semantic:off"


def test_scorer_computes_the_three_features():
    pipe = DifficultyPipeline(semantic=hash_scorer(), semantic_status="minilm:deadbeef")
    resp = pipe.run(DECK, "deck.pptx", GRAPH)
    assert resp.predictions
    for p in resp.predictions:
        d, e, s = (feat(p, n) for n in SEMANTIC)
        assert all(f.status is FeatureStatus.COMPUTED for f in (d, e, s))
        assert all(f.source == "semantic:minilm" for f in (d, e, s))
        assert s.value == (min(d.value, 1) + min(e.value, 1)) / 2
    assert not [w for w in resp.warnings if "semantic" in w]
    assert any(feat(p, "expl.semantic_definition_count").value for p in resp.predictions)


def test_unavailable_scorer_status_is_reported_per_feature():
    pipe = DifficultyPipeline(semantic=None, semantic_status="minilm:unavailable")
    resp = pipe.run(DECK, "deck.pptx", GRAPH)
    f = feat(resp.predictions[0], "expl.semantic_support_score")
    assert (f.status, f.value, f.source) == (FeatureStatus.UNAVAILABLE, None, "semantic:minilm")
    assert f.note == "semantic scorer unavailable"


def test_repeated_runs_are_identical():
    pipe = DifficultyPipeline(semantic=hash_scorer(), semantic_status="minilm:deadbeef")
    first = pipe.run(DECK, "deck.pptx", GRAPH).model_dump_json()
    assert pipe.run(DECK, "deck.pptx", GRAPH).model_dump_json() == first


def test_sentence_budget_warns_once_and_is_per_request():
    pipe = DifficultyPipeline(
        semantic=hash_scorer(max_sentences=3), semantic_status="minilm:deadbeef"
    )
    resp = pipe.run(DECK, "deck.pptx", GRAPH)
    assert resp.warnings.count(SEMANTIC_TRUNCATED_WARNING) == 1
    again = pipe.run(DECK, "deck.pptx", GRAPH)  # the budget starts over for the next request
    assert again.model_dump_json() == resp.model_dump_json()
    assert any(
        feat(p, "expl.semantic_definition_count").note.startswith("sentence budget")
        for p in resp.predictions
    )


def test_no_truncation_warning_with_the_default_budget():
    pipe = DifficultyPipeline(semantic=hash_scorer(), semantic_status="minilm:deadbeef")
    assert SEMANTIC_TRUNCATED_WARNING not in pipe.run(DECK, "deck.pptx", GRAPH).warnings


# ------------------------------------------------------------------ baseline guard


def test_support_features_are_the_three_regex_ones():
    assert SUPPORT_FEATURES == ("expl.definition_count", "expl.example_count", "expl.step_count")
    assert not set(SUPPORT_FEATURES) & set(SEMANTIC)
    assert all(
        s.requires == "text" and s.maturity == "implemented"
        for s in FEATURE_SPECS
        if s.name in SUPPORT_FEATURES
    )


@pytest.mark.parametrize("kind", ["pptx", "pdf"])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_baseline_prediction_ignores_the_scorer(n, kind):
    lecture_id = f"synthetic-lecture-{n}"
    spec = make_deck_spec(42 + n, 8, lecture_id=lecture_id)
    graph = C1GraphResponse.model_validate(build_c1_graph(spec, "v1"))
    data = build_pptx(spec) if kind == "pptx" else build_pdf(spec)
    off = DifficultyPipeline().run(data, f"deck.{kind}", graph)
    on = DifficultyPipeline(semantic=hash_scorer(), semantic_status="minilm:deadbeef").run(
        data, f"deck.{kind}", graph
    )
    assert [(p.score, p.difficulty) for p in off.predictions] == [
        (p.score, p.difficulty) for p in on.predictions
    ]
    assert off.predictions
    for a, b in zip(off.predictions, on.predictions, strict=True):
        regex = {f.name: f for f in a.features if f.name not in SEMANTIC}
        assert regex == {f.name: f for f in b.features if f.name not in SEMANTIC}


# ------------------------------------------------------------------ router construction


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(RT, "configure_logging", lambda level: None)
    for key in ("SEMANTIC", "SEMANTIC_MODEL_DIR"):
        monkeypatch.delenv(f"EDUSYNC_C2_{key}", raising=False)
    get_c2_settings.cache_clear()
    get_pipeline.cache_clear()
    yield
    get_c2_settings.cache_clear()
    get_pipeline.cache_clear()


def test_router_off_never_touches_the_loader(fresh, monkeypatch):
    def boom(model_dir):
        raise AssertionError("the loader must not run when the scorer is off")

    monkeypatch.setattr(RT, "get_default_encoder", boom)
    pipe = get_pipeline()
    assert (pipe.semantic, pipe.semantic_status) == (None, "off")


def test_router_builds_the_scorer_from_settings(fresh, monkeypatch, keyword_encoder, tmp_path):
    seen = []
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC", "minilm")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_DEF_THRESHOLD", "0.4")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_EX_THRESHOLD", "0.5")
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC_MAX_SENTENCES", "9")
    monkeypatch.setattr(
        RT, "get_default_encoder", lambda model_dir: seen.append(model_dir) or keyword_encoder
    )
    pipe = get_pipeline()
    assert seen == [tmp_path]
    assert pipe.semantic_status == "minilm:abcd1234"
    scorer = pipe.semantic
    assert (scorer.def_threshold, scorer.ex_threshold, scorer.max_sentences) == (0.4, 0.5, 9)
    assert get_pipeline() is pipe
    assert seen == [tmp_path]  # lru_cache: built once


def test_router_unavailable_encoder(fresh, monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC", "minilm")
    monkeypatch.setattr(RT, "get_default_encoder", lambda model_dir: None)
    pipe = get_pipeline()
    assert (pipe.semantic, pipe.semantic_status) == (None, "minilm:unavailable")


def test_router_minilm_without_a_dir_is_unavailable(fresh, monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_SEMANTIC", "minilm")
    pipe = get_pipeline()
    assert (pipe.semantic, pipe.semantic_status) == (None, "minilm:unavailable")


def test_prototype_classes_match_the_scorer():
    assert set(load_prototypes()) == set(CLASSES)
