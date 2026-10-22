"""Phase 2 integration: all 35 features end to end on the seeded synthetic decks.

Runs the whole pipeline (parse, graph, concepts, features, baseline) over seeded PPTX and PDF
decks, plain and with varied layout, under four setups: the default, the heuristic concept
filter, a fake semantic scorer and the C1 API provider behind a fake transport. Nothing here
needs the network or model weights. The checks are structural (registry order, no mocked or
planned leftovers, honest unavailable notes, determinism); they do not claim the values are right.
"""

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.concept_filter import HeuristicConceptFilter
from edusync_c2.difficulty.providers import C1ApiGraphProvider
from edusync_c2.difficulty.registry import FEATURE_NAMES, FEATURE_SPECS
from edusync_c2.difficulty.schemas import (
    C1GraphResponse,
    DifficultyResponse,
    FeatureStatus,
)
from edusync_c2.difficulty.synthetic import (
    build_c1_graph,
    build_pdf,
    build_pptx,
    make_deck_spec,
)

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
SEEDS = (42, 43, 44)
SEMANTIC_FEATURES = {
    "expl.semantic_definition_count",
    "expl.semantic_example_count",
    "expl.semantic_support_score",
}
# Features that read the C1 graph (group B) or its frequency; mocked when the graph is a mock.
MOCK_GRAPH_FEATURES = {
    "dens.concept_frequency",
    "prereq.chain_depth",
    "prereq.curriculum_defined_share",
    "prereq.dependents_count",
    "prereq.direct_count",
    "prereq.mean_confidence",
    "prereq.transitive_count",
    "prereq.weighted_depth",
}
PPTX_STRUCTURE_FEATURES = {s.name for s in FEATURE_SPECS if s.requires == "pptx_structure"}
LAYOUT_FEATURES = (
    "dens.text_area_ratio",
    "dens.picture_area_ratio",
    "dens.min_font_pt",
    "dens.diagram_element_count",
    "dens.table_cell_count",
    "dens.visual_complexity",
)

Setup = Callable[[bytes, str, dict, str], DifficultyResponse]


def _spec(seed: int):
    return make_deck_spec(seed, 8, lecture_id=f"synthetic-lecture-{seed - 42}")


def _decks():
    """(id, kind, vary, data, v1 graph dict) for every seed: PPTX, varied PPTX, PDF."""
    for seed in SEEDS:
        spec = _spec(seed)
        graph = build_c1_graph(spec, "v1")
        yield f"{seed}-pptx", "pptx", False, build_pptx(spec), graph
        yield f"{seed}-pptx-varied", "pptx", True, build_pptx(spec, vary_layout=True), graph
        yield f"{seed}-pdf", "pdf", False, build_pdf(spec), graph


DECKS = list(_decks())
DECK_IDS = [d[0] for d in DECKS]


def _default(data: bytes, name: str, graph: dict, _kind: str) -> DifficultyResponse:
    return DifficultyPipeline().run(data, name, C1GraphResponse.model_validate(graph))


def _heuristic(data: bytes, name: str, graph: dict, _kind: str) -> DifficultyResponse:
    pipeline = DifficultyPipeline(concept_filter=HeuristicConceptFilter())
    return pipeline.run(data, name, C1GraphResponse.model_validate(graph))


def _semantic(scorer):
    def run(data: bytes, name: str, graph: dict, _kind: str) -> DifficultyResponse:
        pipeline = DifficultyPipeline(semantic=scorer, semantic_status="minilm:abcd1234")
        return pipeline.run(data, name, C1GraphResponse.model_validate(graph))

    return run


def _c1_api(data: bytes, name: str, graph: dict, _kind: str) -> DifficultyResponse:
    pipeline = DifficultyPipeline(
        graph_provider_factory=lambda d, f: C1ApiGraphProvider(d, f, lambda _b, _n: graph),
        graph_provider_name="c1_api",
    )
    return pipeline.run(data, name)


@pytest.fixture
def setups(make_scorer) -> dict[str, Setup]:
    return {
        "default": _default,
        "heuristic": _heuristic,
        "semantic": _semantic(make_scorer()),
        "c1_api": _c1_api,
    }


SETUP_NAMES = ["default", "heuristic", "semantic", "c1_api"]


def _all_features(response: DifficultyResponse):
    return [(p, f) for p in response.predictions for f in p.features]


def _check_structure(response: DifficultyResponse, kind: str, *, mock_graph: bool) -> None:
    assert response.predictions
    mocked: set[str] = set()
    for prediction in response.predictions:
        assert [f.name for f in prediction.features] == list(FEATURE_NAMES)
        assert len(prediction.features) == 35
        for f in prediction.features:
            assert not f.source.startswith("todo:"), f.name
            assert f.status is not None
            if f.status is FeatureStatus.MOCKED:
                mocked.add(f.name)
            elif f.status is FeatureStatus.UNAVAILABLE:
                assert f.value is None
                assert f.note.strip(), f"{f.name} is unavailable without a note"
            else:
                assert f.value is not None
    # An implemented feature is never mocked, except the graph-dependent ones on a mock graph.
    assert mocked == (MOCK_GRAPH_FEATURES if mock_graph else set())
    if kind == "pdf":
        for _, f in _all_features(response):
            if f.name in PPTX_STRUCTURE_FEATURES:
                assert f.status is FeatureStatus.UNAVAILABLE, f.name


@pytest.mark.parametrize("setup", SETUP_NAMES)
@pytest.mark.parametrize("deck", DECKS, ids=DECK_IDS)
def test_every_setup_yields_all_features_honestly(deck, setup, setups) -> None:
    _, kind, _, data, graph = deck
    response = setups[setup](data, f"lecture.{kind}", graph, kind)
    _check_structure(response, kind, mock_graph=False)
    names = {f.name: f for _, f in _all_features(response)}
    assert len(names) == 35


@pytest.mark.parametrize(
    "kind_deck", [d for d in DECKS if d[0].startswith("42") and d[1] == "pptx"], ids=lambda d: d[0]
)
def test_mock_graph_marks_exactly_the_graph_features(kind_deck) -> None:
    # The slide-title mock needs slide titles, so it has no concepts for a PDF.
    _, kind, _, data, _ = kind_deck
    response = DifficultyPipeline().run(data, f"lecture.{kind}")
    assert response.inputs[1].kind == "mock"
    _check_structure(response, kind, mock_graph=True)


@pytest.mark.parametrize("deck", DECKS, ids=DECK_IDS)
def test_semantic_features_follow_the_scorer(deck, setups) -> None:
    _, kind, _, data, graph = deck
    off = setups["default"](data, f"lecture.{kind}", graph, kind)
    on = setups["semantic"](data, f"lecture.{kind}", graph, kind)
    for _, f in _all_features(off):
        if f.name in SEMANTIC_FEATURES:
            assert f.status is FeatureStatus.UNAVAILABLE
            assert f.source == "semantic:off"
    for _, f in _all_features(on):
        if f.name in SEMANTIC_FEATURES:
            assert f.status is FeatureStatus.COMPUTED
            assert f.source == "semantic:minilm"


@pytest.mark.parametrize("deck", DECKS, ids=DECK_IDS)
def test_baseline_is_identical_with_and_without_semantic(deck, setups) -> None:
    _, kind, _, data, graph = deck
    off = setups["default"](data, f"lecture.{kind}", graph, kind)
    on = setups["semantic"](data, f"lecture.{kind}", graph, kind)
    assert [(p.concept_id, p.difficulty, p.score) for p in off.predictions] == [
        (p.concept_id, p.difficulty, p.score) for p in on.predictions
    ]
    # Every other feature value is untouched by the scorer.
    for (_, a), (_, b) in zip(_all_features(off), _all_features(on), strict=True):
        if a.name not in SEMANTIC_FEATURES:
            assert (a.value, a.status, a.source) == (b.value, b.status, b.source), a.name


@pytest.mark.parametrize("deck", DECKS, ids=DECK_IDS)
def test_other_setups_keep_the_baseline_on_synthetic_decks(deck, setups) -> None:
    # On the seeded decks the heuristic filter drops and merges nothing, and a C1 API graph equal
    # to the payload graph gives the same scores; real decks may differ (heuristic is opt-in).
    _, kind, _, data, graph = deck
    base = setups["default"](data, f"lecture.{kind}", graph, kind)
    want = [(p.concept_id, p.difficulty, p.score) for p in base.predictions]
    for name in ("heuristic", "c1_api"):
        other = setups[name](data, f"lecture.{kind}", graph, kind)
        assert [(p.concept_id, p.difficulty, p.score) for p in other.predictions] == want, name


def test_c1_api_setup_is_labelled_real_with_a_derived_id(setups) -> None:
    _, kind, _, data, graph = DECKS[0]
    response = setups["c1_api"](data, f"lecture.{kind}", graph, kind)
    assert response.inputs[1].kind == "real"
    assert str(response.lecture_id).startswith("c1-")
    assert not any("mock" in w for w in response.warnings)


@pytest.mark.parametrize("deck", [d for d in DECKS if d[2]], ids=lambda d: d[0])
def test_varied_layout_decks_make_group_d_vary(deck) -> None:
    _, kind, _, data, graph = deck
    response = _default(data, f"lecture.{kind}", graph, kind)
    for name in LAYOUT_FEATURES:
        values = {
            f.value
            for _, f in _all_features(response)
            if f.name == name and f.status is FeatureStatus.COMPUTED
        }
        assert len(values) > 1, name


@pytest.mark.parametrize("setup", SETUP_NAMES)
@pytest.mark.parametrize("deck", DECKS[:3], ids=DECK_IDS[:3])
def test_two_runs_are_identical(deck, setup, setups) -> None:
    _, kind, _, data, graph = deck
    first = setups[setup](data, f"lecture.{kind}", graph, kind)
    second = setups[setup](data, f"lecture.{kind}", graph, kind)
    assert first.model_dump_json() == second.model_dump_json()


CHILD = (
    "import sys\n"
    "from edusync_c2.difficulty import DifficultyPipeline\n"
    "from edusync_c2.difficulty.concept_filter import HeuristicConceptFilter\n"
    "from edusync_c2.difficulty.schemas import C1GraphResponse\n"
    "from edusync_c2.difficulty.synthetic import build_c1_graph, build_pdf, build_pptx,"
    " make_deck_spec\n"
    "spec = make_deck_spec(43, 8, lecture_id='synthetic-lecture-1')\n"
    "graph = C1GraphResponse.model_validate(build_c1_graph(spec, 'v1'))\n"
    "out = []\n"
    "decks = ((build_pptx(spec, vary_layout=True), 'a.pptx'), (build_pdf(spec), 'a.pdf'))\n"
    "for data, name in decks:\n"
    "    for pipe in (DifficultyPipeline(), DifficultyPipeline(concept_filter="
    "HeuristicConceptFilter())):\n"
    "        out.append(pipe.run(data, name, graph).model_dump_json())\n"
    "sys.stdout.write('\\n'.join(out))\n"
)


def _run_child(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), *sys.path])
    done = subprocess.run(
        [sys.executable, "-c", CHILD],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )
    return done.stdout


def test_json_response_is_identical_under_different_hash_seeds() -> None:
    first, second = _run_child("0"), _run_child("1")
    assert first == second
    lines = first.splitlines()
    assert len(lines) == 4
    assert all(len(json.loads(line)["predictions"][0]["features"]) == 35 for line in lines)


# The feature_report 'check' set on varied decks, with the reason each is expected to be flagged:
#   dens.concepts_per_slide: the generator puts exactly one concept on every slide, so it is
#       constant 1.0 by construction (real decks vary).
#   expl.semantic_*: the report runs the default pipeline, where the semantic scorer is off, so
#       the three features are unavailable on purpose (they are computed when it is on).
EXPECTED_CHECKS = {
    "dens.concepts_per_slide",
    "expl.semantic_definition_count",
    "expl.semantic_example_count",
    "expl.semantic_support_score",
}


def test_feature_report_check_set_on_varied_decks() -> None:
    module_spec = importlib.util.spec_from_file_location(
        "feature_report_m6", SCRIPTS / "feature_report.py"
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    report = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(report)
    rows, predictions = report.collect(42, 3, 8, vary_layout=True)
    assert predictions == 48
    assert set(report.checks(rows)) == EXPECTED_CHECKS
    by_name = {r["name"]: r for r in rows}
    # Varied layout makes every group D structure feature non-constant on PPTX.
    for name in LAYOUT_FEATURES:
        assert by_name[name]["distinct"] > 1, name
