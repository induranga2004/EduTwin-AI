"""Shared fixtures: a tiny synthetic lecture on disk and loaders for the C2 scripts."""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from edusync_c2.difficulty.synthetic import build_c1_graph, build_pptx, make_deck_spec
from edusync_c2_model.manifest import ManifestEntry

C2_SCRIPTS = Path(__file__).resolve().parents[2] / "c2" / "scripts"
PACKAGE_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def c2_script() -> Callable[[str], ModuleType]:
    return lambda stem: load_script(C2_SCRIPTS / f"{stem}.py", f"c2_script_{stem}")


@pytest.fixture
def package_script() -> Callable[[str], ModuleType]:
    """Load a fresh copy of one of this package's scripts (so tests can patch it)."""
    return lambda stem: load_script(PACKAGE_SCRIPTS / f"{stem}.py", f"package_script_{stem}")


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


@pytest.fixture(scope="session")
def toy() -> ModuleType:
    """The ``toydata`` helpers (toy Examples and the hash-based fake embedder)."""
    return load_script(Path(__file__).resolve().parent / "toydata.py", "toydata")
