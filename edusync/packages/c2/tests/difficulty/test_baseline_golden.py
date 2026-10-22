"""Guard: the rule baseline and the established features do not drift when features are added.

``golden/baseline_synthetic.json`` holds the labels, scores and feature values of the seeded
synthetic decks (PPTX and PDF, v1 graphs) as produced before Feature A became real. Features
added later are ignored here; ``lang.unfamiliar_vocab_ratio`` is excluded because it changes
from the mock to a real value. Regenerate only when a milestone intentionally changes them.
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
    response = _run(key)
    expected = GOLDEN[key]
    assert [p.concept_id for p in response.predictions] == [e["concept_id"] for e in expected]
    for prediction, want in zip(response.predictions, expected, strict=True):
        got_difficulty = prediction.difficulty.value if prediction.difficulty else None
        assert got_difficulty == want["difficulty"]
        assert prediction.score == want["score"]
        by_name = {f.name: f for f in prediction.features}
        for name, (value, status) in want["features"].items():
            assert by_name[name].value == value, name
            assert by_name[name].status.value == status, name
