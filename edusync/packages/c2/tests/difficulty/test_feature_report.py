"""Tests for the feature report CLI."""

import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from edusync_c2.difficulty.registry import FEATURE_SPECS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "feature_report.py"
SNAPSHOT = Path(__file__).parent / "golden" / "registry_snapshot.json"
ARGS = ["--lectures", "1", "--concepts", "4"]


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("feature_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def collected(script):
    return script.collect(42, 1, 4)


def test_text_report_lists_all_features_and_flags(script, collected) -> None:
    rows, predictions = collected
    text = script.format_report(rows, predictions, 42, 1)
    assert len(FEATURE_SPECS) == 35
    lines = text.splitlines()
    for spec in FEATURE_SPECS:
        line = next(x for x in lines if spec.name in x)
        if spec.maturity == "planned":
            assert "planned" in line
    visual = next(x for x in lines if "dens.visual_complexity" in x)
    assert "planned" not in visual  # implemented since Feature D
    assert "[A_language]" in text
    assert "check" in text


def test_counts_sum_to_predictions(collected) -> None:
    rows, predictions = collected
    assert len(rows) == 35
    assert predictions == 8  # 4 concepts x (pptx, pdf)
    for row in rows:
        assert row["computed"] + row["mocked"] + row["unavailable"] == predictions


def test_check_excludes_planned(script, collected) -> None:
    rows, _ = collected
    for name in script.checks(rows):
        row = next(r for r in rows if r["name"] == name)
        assert "planned" not in row["flags"]


def test_check_footer_none(script) -> None:
    assert script.format_report([], 0, 1, 1).endswith("check: none")


def test_main_text_exit_zero(script, capsys) -> None:
    assert script.main(ARGS) == 0
    assert "Feature report" in capsys.readouterr().out


def test_main_json(script, capsys) -> None:
    assert script.main([*ARGS, "--json"]) == 0
    out = capsys.readouterr().out
    assert out.endswith("}\n")
    assert "\r" not in out
    data = json.loads(out)
    assert len(data["features"]) == 35
    assert list(data) == sorted(data)
    for row in data["features"]:
        assert row["computed"] + row["mocked"] + row["unavailable"] == data["predictions"]


def test_deterministic(script, capsys) -> None:
    script.main(ARGS)
    first = capsys.readouterr().out
    script.main(ARGS)
    assert capsys.readouterr().out == first


def test_write_snapshot_matches_committed(script, tmp_path: Path) -> None:
    out = tmp_path / "snap.json"
    assert script.main(["--write-snapshot", str(out)]) == 0
    assert out.read_bytes() == SNAPSHOT.read_bytes()


def test_bad_snapshot_path_exits_two(script, tmp_path: Path, capsys) -> None:
    assert script.main(["--write-snapshot", str(tmp_path / "no" / "dir" / "s.json")]) == 2
    assert "Cannot write" in capsys.readouterr().err


def test_bad_arguments_exit_two(script) -> None:
    assert script.main(["--lectures", "x"]) == 2
    assert script.main(["--lectures", "0"]) == 2


def test_help_exits_zero(script, capsys) -> None:
    assert script.main(["--help"]) == 0
    assert "synthetic decks" in capsys.readouterr().out.lower()


def test_runs_as_main(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "snap.json"
    monkeypatch.setattr("sys.argv", ["feature_report.py", "--write-snapshot", str(out)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert out.read_bytes() == SNAPSHOT.read_bytes()
