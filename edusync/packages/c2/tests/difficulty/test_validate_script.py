"""Tests for the validate_c1_graph CLI."""

import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from edusync_c2.difficulty.schemas import C1GraphResponse
from edusync_c2.difficulty.synthetic import build_c1_graph, make_deck_spec

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "validate_c1_graph.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("validate_c1_graph", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _slugify(data: dict) -> dict:
    for node in data["graph"]["nodes"]:
        node["id"] = node["id"].replace(" ", "-")
    for edge in data["graph"]["edges"]:
        edge["source"] = edge["source"].replace(" ", "-")
        edge["target"] = edge["target"].replace(" ", "-")
    return data


def _write(tmp_path: Path, data: object, name: str = "graph.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_valid_v1_human_report(script, tmp_path: Path, capsys, graph_v1: C1GraphResponse) -> None:
    path = _write(tmp_path, _slugify(graph_v1.model_dump(mode="json")))
    assert script.main(["--file", str(path)]) == 0
    out = capsys.readouterr().out
    assert "Detected version: v1" in out
    assert "Result: ok" in out
    assert "0 error(s)" in out


def test_valid_synthetic_v0_and_v1(script, tmp_path: Path, capsys) -> None:
    spec = make_deck_spec(3, 6)
    for version in ("v0", "v1"):
        data = build_c1_graph(spec, version)
        path = _write(tmp_path, _slugify(data) if version == "v1" else data, f"{version}.json")
        assert script.main(["--file", str(path), "--expected", version]) == 0
    assert "Detected version: v0" in capsys.readouterr().out


def test_errors_exit_one_and_group_findings(script, tmp_path: Path, capsys) -> None:
    data = build_c1_graph(make_deck_spec(3, 6), "v1")
    data["graph"]["edges"][0]["relation_type"] = "bogus"
    data["graph"]["node_count"] = 99  # warning
    data["graph"]["edges"].append({"source": "x", "target": "y", "relation_type": "co_mentioned"})
    assert script.main(["--file", str(_write(tmp_path, data))]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert out.index("ERROR C1-010") < out.index("WARNING C1-005")
    assert "[graph.edges[0]]" in out


def test_finding_without_location(script, tmp_path: Path, capsys) -> None:
    assert script.main(["--file", str(_write(tmp_path, [1, 2]))]) == 1
    assert "ERROR C1-001: " in capsys.readouterr().out


def test_json_output(script, tmp_path: Path, capsys, graph_v0: C1GraphResponse) -> None:
    path = _write(tmp_path, graph_v0.model_dump(mode="json"))
    assert script.main(["--file", str(path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["version"] == "v0"
    assert report["ok"] is True
    assert report["stats"]["nodes"] == 4


def test_expected_flag_changes_expectations(script, tmp_path: Path, capsys, graph_v0) -> None:
    path = _write(tmp_path, graph_v0.model_dump(mode="json"))
    assert script.main(["--file", str(path), "--expected", "v1", "--json"]) == 1
    rules = {f["rule"] for f in json.loads(capsys.readouterr().out)["findings"]}
    assert "C1-018" in rules  # ids with spaces are not slugs


def test_missing_file_exits_two(script, tmp_path: Path, capsys) -> None:
    assert script.main(["--file", str(tmp_path / "nope.json")]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_invalid_json_exits_two(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    assert script.main(["--file", str(path)]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_non_ascii_output_is_utf8(script, tmp_path: Path, capsys) -> None:
    data = {"graph": {"nodes": [{"id": "É", "label": "x"}], "edges": []}}
    assert script.main(["--file", str(_write(tmp_path, data)), "--json"]) == 0
    assert "É" in capsys.readouterr().out


def test_runs_as_main(tmp_path: Path, monkeypatch, graph_v0) -> None:
    path = _write(tmp_path, graph_v0.model_dump(mode="json"))
    monkeypatch.setattr("sys.argv", ["validate_c1_graph.py", "--file", str(path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
