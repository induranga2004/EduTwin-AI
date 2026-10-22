"""Shared fixtures: a tiny synthetic lecture on disk and loaders for the C2 scripts."""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from edusync_c2.difficulty.synthetic import build_c1_graph, build_pptx, make_deck_spec
from edusync_c2_model.manifest import ManifestEntry

sys.path.insert(0, str(Path(__file__).parent))  # lets tests import public_fixtures


@pytest.fixture
def make_entry(tmp_path: Path) -> Callable[..., ManifestEntry]:
    """Write a synthetic deck (and optionally its v1 C1 graph) and return its manifest entry."""

    def make(seed: int = 7, *, graph: bool = True, lecture_id: str = "L001", course: str = "C1"):
        spec = make_deck_spec(seed, n_concepts=4, lecture_id=lecture_id)
        folder = tmp_path / f"lec-{lecture_id}"
        folder.mkdir(exist_ok=True)
        file = folder / f"{lecture_id}.pptx"
        file.write_bytes(build_pptx(spec))
        graph_path = None
        if graph:
            graph_path = folder / f"{lecture_id}.json"
            graph_path.write_text(json.dumps(build_c1_graph(spec, "v1")), encoding="utf-8")
        return ManifestEntry(lecture_id, course, file, graph_path)

    return make
