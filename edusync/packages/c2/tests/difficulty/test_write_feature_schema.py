"""Tests for the write_feature_schema CLI."""

import importlib.util
import json
import runpy
from pathlib import Path

import pytest

from edusync_c2.difficulty.models import OnnxFusionModel
from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_hash,
)

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "write_feature_schema.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("write_feature_schema", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_writes_registry_payload(script, tmp_path: Path, capsys) -> None:
    out = tmp_path / "feature_schema.json"
    assert script.main(["--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "names": list(FEATURE_NAMES),
        "hash": feature_schema_hash(),
    }
    assert "Wrote" in capsys.readouterr().out


def test_format_is_indent_one_lf_with_trailing_newline(script, tmp_path: Path) -> None:
    out = tmp_path / "feature_schema.json"
    assert script.main(["--out", str(out)]) == 0
    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"}\n")
    assert raw.decode("utf-8") == json.dumps(json.loads(raw), indent=1) + "\n"
    assert list(json.loads(raw)) == ["schema_version", "names", "hash"]


def test_creates_missing_parent_directory(script, tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "feature_schema.json"
    assert script.main(["--out", str(out)]) == 0
    assert out.is_file()


def test_default_out_is_cwd(script, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert script.main([]) == 0
    assert (tmp_path / "feature_schema.json").is_file()


def test_unwritable_path_exits_two(script, tmp_path: Path, capsys) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert script.main(["--out", str(blocker / "feature_schema.json")]) == 2
    assert "Cannot write" in capsys.readouterr().err


def test_bad_argument_exits_two(script, capsys) -> None:
    assert script.main(["--nope"]) == 2


def test_help_exits_zero(script, capsys) -> None:
    assert script.main(["--help"]) == 0


def test_check_accepts_a_fresh_file(script, tmp_path: Path, capsys) -> None:
    out = tmp_path / "feature_schema.json"
    script.main(["--out", str(out)])
    capsys.readouterr()
    assert script.main(["--check", str(out)]) == 0
    assert "matches the registry" in capsys.readouterr().out


def test_check_does_not_write(script, tmp_path: Path) -> None:
    assert script.main(["--check", str(tmp_path / "missing.json")]) == 2
    assert not (tmp_path / "feature_schema.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "0.0"), ("hash", "deadbeef")],
)
def test_check_names_the_differing_scalar_field(
    script, tmp_path: Path, capsys, field: str, value: str
) -> None:
    payload = script.schema_payload()
    payload[field] = value
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert script.main(["--check", str(path)]) == 1
    out = capsys.readouterr().out
    assert f"MISMATCH {field} differs" in out
    assert out.count("MISMATCH") == 1


def test_check_reports_missing_and_unexpected_names(script, tmp_path: Path, capsys) -> None:
    payload = script.schema_payload()
    payload["names"] = [*FEATURE_NAMES[1:], "bogus.feature"]
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert script.main(["--check", str(path)]) == 1
    out = capsys.readouterr().out
    assert "names differs (missing" in out
    assert FEATURE_NAMES[0] in out
    assert "bogus.feature" in out


def test_check_reports_reordered_names(script, tmp_path: Path, capsys) -> None:
    payload = script.schema_payload()
    payload["names"] = list(reversed(FEATURE_NAMES))
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert script.main(["--check", str(path)]) == 1
    assert "different order" in capsys.readouterr().out


def test_check_reports_names_that_are_not_a_list(script, tmp_path: Path, capsys) -> None:
    payload = script.schema_payload()
    payload["names"] = "nope"
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert script.main(["--check", str(path)]) == 1
    assert "not a list" in capsys.readouterr().out


def test_check_reports_every_missing_field(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "schema.json"
    path.write_text("{}", encoding="utf-8")
    assert script.main(["--check", str(path)]) == 1
    assert capsys.readouterr().out.count("MISMATCH") == 3


def test_check_rejects_a_non_object(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "schema.json"
    path.write_text("[1]", encoding="utf-8")
    assert script.main(["--check", str(path)]) == 1
    assert "not a JSON object" in capsys.readouterr().out


def test_check_invalid_json_exits_two(script, tmp_path: Path, capsys) -> None:
    path = tmp_path / "schema.json"
    path.write_text("{oops", encoding="utf-8")
    assert script.main(["--check", str(path)]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_round_trip_with_the_artifact_validator(script, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_int8.onnx").write_bytes(b"int8")
    (model_dir / "labels.json").write_text('["Low", "Medium", "High"]', encoding="utf-8")
    (model_dir / "inference_config.json").write_text(
        json.dumps({"max_length": 256, "base_model": "m", "base_revision": "r"}), encoding="utf-8"
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert script.main(["--out", str(model_dir / "feature_schema.json")]) == 0
    model = OnnxFusionModel.from_artifacts(model_dir)
    assert model.info.version == FEATURE_SCHEMA_VERSION


def test_runs_as_main(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "feature_schema.json"
    monkeypatch.setattr("sys.argv", ["write_feature_schema.py", "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert exc.value.code == 0
    assert out.is_file()
