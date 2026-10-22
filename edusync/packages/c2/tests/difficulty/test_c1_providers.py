"""C1ApiGraphProvider (fake transport) and FileGraphProvider (bundled sample and custom files)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from edusync_c2.difficulty.contract_check import check_c1_graph
from edusync_c2.difficulty.errors import GraphProviderError
from edusync_c2.difficulty.providers import C1ApiGraphProvider, FileGraphProvider
from edusync_c2.difficulty.schemas import MaterialsDoc
from edusync_c2.difficulty.synthetic import build_c1_graph, make_deck_spec

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_sample_graph.py"
SAMPLE = (
    Path(__file__).resolve().parents[2]
    / "src/edusync_c2/difficulty/resources/sample_c1_graph_v1.json"
)


def _graph(version: str, lecture_id: str = "c1-uuid-1") -> dict:
    return build_c1_graph(make_deck_spec(3, 4, lecture_id=lecture_id), version)  # type: ignore[arg-type]


@pytest.mark.parametrize("version", ["v0", "v1"])
def test_c1_api_valid_graph(deck: MaterialsDoc, version: str) -> None:
    graph = _graph(version)
    calls: list[tuple[bytes, str]] = []

    def transport(data: bytes, filename: str) -> dict:
        calls.append((data, filename))
        return graph

    ctx = C1ApiGraphProvider(b"BYTES", "deck.pptx", transport).build(deck)
    assert calls == [(b"BYTES", "deck.pptx")]
    assert C1ApiGraphProvider.name == "c1_api"
    assert (ctx.source.name, ctx.source.kind) == ("c1_graph", "real")
    assert ctx.source.detail == (
        f"C1 API graph ({len(graph['graph']['nodes'])} nodes, {len(graph['graph']['edges'])} edges)"
    )
    assert ctx.response.lecture_id == "c1-" + hashlib.sha256(b"BYTES").hexdigest()[:12]
    assert ctx.response.lecture_id != graph["lecture_id"]
    assert len(ctx.response.graph.nodes) == len(graph["graph"]["nodes"])


def test_c1_api_id_ignores_c1_uuid_and_depends_on_bytes(deck: MaterialsDoc) -> None:
    def run(data: bytes, lecture_id: str) -> str | None:
        provider = C1ApiGraphProvider(data, "d.pdf", lambda d, f: _graph("v1", lecture_id))
        return provider.build(deck).response.lecture_id

    assert run(b"same", "uuid-a") == run(b"same", "uuid-b")
    assert run(b"same", "uuid-a") != run(b"other", "uuid-a")


@pytest.mark.parametrize("bad", [{}, {"graph": "nope"}, {"graph": {"nodes": [{"id": 1}]}}])
def test_c1_api_invalid_shape_is_502(deck: MaterialsDoc, bad: dict) -> None:
    with pytest.raises(GraphProviderError) as exc:
        C1ApiGraphProvider(b"x", "d.pdf", lambda d, f: bad).build(deck)
    assert (exc.value.status_code, exc.value.message) == (502, "C1 returned an invalid graph")


def test_c1_api_transport_error_propagates(deck: MaterialsDoc) -> None:
    def transport(data: bytes, filename: str) -> dict:
        raise GraphProviderError(504, "C1 timed out")

    with pytest.raises(GraphProviderError) as exc:
        C1ApiGraphProvider(b"x", "d.pdf", transport).build(deck)
    assert exc.value.status_code == 504


def test_file_provider_default_sample(deck: MaterialsDoc) -> None:
    ctx = FileGraphProvider().build(deck)
    assert FileGraphProvider.name == "file"
    assert (ctx.source.name, ctx.source.kind) == ("c1_graph", "mock")
    assert ctx.source.detail == "graph file: sample_c1_graph_v1.json"
    assert ctx.response.lecture_id == "sample-lecture"


def test_bundled_sample_passes_contract() -> None:
    report = check_c1_graph(json.loads(SAMPLE.read_text(encoding="utf-8")))
    assert report.version == "v1"
    assert report.ok
    assert report.findings == []
    assert report.stats["prerequisite_edges"] > 0
    assert b"\r" not in SAMPLE.read_bytes()


def test_sample_regeneration_is_byte_identical(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_sample_graph", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "nested" / "g.json"
    assert module.main(["--out", str(out)]) == 0
    assert out.read_bytes() == SAMPLE.read_bytes()
    assert module.DEFAULT_OUT.resolve() == SAMPLE.resolve()


def test_file_provider_custom_path(deck: MaterialsDoc, tmp_path: Path) -> None:
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(_graph("v1", "keep-me")), encoding="utf-8")
    ctx = FileGraphProvider(path).build(deck)
    assert ctx.source.detail == "graph file: mine.json"
    assert ctx.response.lecture_id == "keep-me"


def test_file_provider_missing_file(deck: MaterialsDoc, tmp_path: Path) -> None:
    with pytest.raises(GraphProviderError) as exc:
        FileGraphProvider(tmp_path / "secret-dir" / "nope.json").build(deck)
    assert (exc.value.status_code, exc.value.message) == (500, "graph file could not be loaded")
    assert "secret-dir" not in exc.value.message


@pytest.mark.parametrize("content", ["{not json", '{"graph": "x"}', "[]"])
def test_file_provider_invalid_content(deck: MaterialsDoc, tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(GraphProviderError) as exc:
        FileGraphProvider(path).build(deck)
    assert exc.value.status_code == 500
