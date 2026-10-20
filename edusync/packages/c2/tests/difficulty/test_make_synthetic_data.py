"""Test for the synthetic data CLI wrapper."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_synthetic_data.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("make_synthetic_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writes_all_files(script, tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert (
        script.main(["--seed", "1", "--lectures", "2", "--concepts", "4", "--out", str(out)]) == 0
    )
    expected = {"labels.csv", "README.txt"}
    for k in (1, 2):
        expected |= {f"lecture_{k}.pptx", f"lecture_{k}.pdf"}
        expected |= {f"lecture_{k}_graph_v0.json", f"lecture_{k}_graph_v1.json"}
    assert {p.name for p in out.iterdir()} == expected
    text = (out / "lecture_1_graph_v1.json").read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text)["graph"]["node_count"] == 4
    assert "circular" in (out / "README.txt").read_text(encoding="utf-8")
    assert len((out / "labels.csv").read_text(encoding="utf-8").splitlines()) == 9


def test_default_out_is_next_to_package(script) -> None:
    assert SCRIPT.parents[1] / "data" / "synthetic" == script.DEFAULT_OUT


@pytest.mark.parametrize("flag", [["--lectures", "0"], ["--concepts", "0"]])
def test_bad_arguments(script, tmp_path: Path, flag: list[str]) -> None:
    assert script.main([*flag, "--out", str(tmp_path)]) == 2
