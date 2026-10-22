"""Guard: the rule baseline and the established features do not drift when features are added.

``golden/baseline_synthetic.json`` holds the labels, scores and feature values of the seeded
synthetic decks (PPTX and PDF, v1 graphs) as produced before Feature A became real. Features
added later are ignored here. Features that intentionally went from mock or planned to real are
excluded explicitly: ``lang.unfamiliar_vocab_ratio`` (never stored), ``EXCLUDED`` (all file
kinds: ``expl.semantic_support_score`` is stored as ``[0.0, "mocked"]`` but is now computed by the
opt-in semantic scorer and unavailable when it is off, which is the default here) and, per file
kind, ``EXCLUDED_BY_KIND`` (``dens.visual_complexity`` is stored as unavailable but is now computed
for PPTX decks; PDF decks still report it unavailable and stay checked). Scores and labels never
change. Regenerate only when a milestone intentionally changes them.
"""

import json
from pathlib import Path

import pytest

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.schemas import C1GraphResponse, DifficultyResponse
from edusync_c2.difficulty.synthetic import (
    build_c1_graph,
    build_pdf,
    build_pptx,
    make_deck_spec,
)

# Stored golden values that changed on purpose, for every file kind (see the docstring).
EXCLUDED: frozenset[str] = frozenset({"expl.semantic_support_score"})
# The same, per file kind.
EXCLUDED_BY_KIND: dict[str, set[str]] = {"pptx": {"dens.visual_complexity"}}

GOLDEN = json.loads(
    (Path(__file__).parent / "golden" / "baseline_synthetic.json").read_text(encoding="utf-8")
)


def _run(key: str) -> DifficultyResponse:
    lecture_id, kind = key.rsplit(".", 1)
    seed = 42 + int(lecture_id.rsplit("-", 1)[1])
    spec = make_deck_spec(seed, 8, lecture_id=lecture_id)
    graph = C1GraphResponse.model_validate(build_c1_graph(spec, "v1"))
    data = build_pptx(spec) if kind == "pptx" else build_pdf(spec)
    return DifficultyPipeline().run(data, f"{lecture_id}.{kind}", graph)


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_baseline_and_established_features_unchanged(key: str) -> None:
    excluded = EXCLUDED | EXCLUDED_BY_KIND.get(key.rsplit(".", 1)[1], set())
    response = _run(key)
    expected = GOLDEN[key]
    assert [p.concept_id for p in response.predictions] == [e["concept_id"] for e in expected]
    for prediction, want in zip(response.predictions, expected, strict=True):
        got_difficulty = prediction.difficulty.value if prediction.difficulty else None
        assert got_difficulty == want["difficulty"]
        assert prediction.score == want["score"]
        by_name = {f.name: f for f in prediction.features}
        for name, (value, status) in want["features"].items():
            if name in excluded:
                continue
            assert by_name[name].value == value, name
            assert by_name[name].status.value == status, name
