import json
from pathlib import Path

import numpy as np
import pytest

from edusync_blueprint_model.config import load_config
from edusync_blueprint_model.onnx_export import check_parity, write_sidecars
from edusync_blueprint_model.quantize import (
    benchmark,
    compare_models,
    file_size_mb,
    format_report,
    write_report,
)


def test_check_parity() -> None:
    a = np.array([[1.0, 2.0]])
    assert check_parity(a, a + 1e-5) == pytest.approx(1e-5)
    with pytest.raises(ValueError, match="differs"):
        check_parity(a, a + 0.1)
    with pytest.raises(ValueError, match="shapes"):
        check_parity(a, np.zeros((2, 2)))
    assert check_parity(np.zeros((0, 2)), np.zeros((0, 2))) == 0.0


def test_write_sidecars(config_file: Path, tmp_path: Path) -> None:
    cfg = load_config(config_file)
    src = tmp_path / "ft"
    (src / "tokenizer").mkdir(parents=True)
    (src / "tokenizer" / "tokenizer.json").write_text("{}")
    (src / "labels.json").write_text(json.dumps({"labels": ["a", "b"]}))
    out = tmp_path / "onnx"
    write_sidecars(cfg, src, out)
    info = json.loads((out / "inference_config.json").read_text())
    assert info["labels"] == ["a", "b"]
    assert info["max_length"] == 16
    assert info["base_revision"] == "abc1234567"
    assert (out / "tokenizer.json").is_file()
    assert (out / "labels.json").is_file()


def test_file_size_mb(tmp_path: Path) -> None:
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * (1024 * 1024))
    assert file_size_mb(f) == pytest.approx(1.0)


def test_benchmark_with_fake_timer() -> None:
    clock = {"t": 0.0}
    calls: list[int] = []

    def timer() -> float:
        clock["t"] += 0.0005  # each call advances 0.5 ms, so a start/stop pair measures 0.5 ms
        return clock["t"]

    stats = benchmark(lambda: calls.append(1), runs=4, timer=timer, warmup=2)
    assert len(calls) == 6
    assert stats.p50_ms == pytest.approx(0.5)
    assert stats.p95_ms == pytest.approx(0.5)
    assert stats.mean_ms == pytest.approx(0.5)


def _fake_predictors():
    def fp32(texts):
        return np.array([[1.0, 0.0] if "a" in t else [0.0, 1.0] for t in texts])

    def int8(texts):
        return np.array([[1.0, 0.0] if "a" in t and "z" not in t else [0.0, 1.0] for t in texts])

    return fp32, int8


def test_compare_format_and_write(tmp_path: Path) -> None:
    fp32_file, int8_file = tmp_path / "f.onnx", tmp_path / "i.onnx"
    fp32_file.write_bytes(b"x" * 4096)
    int8_file.write_bytes(b"x" * 1024)
    fp32, int8 = _fake_predictors()
    texts = ["a1", "b1", "a2", "az"]
    report = compare_models(
        fp32,
        int8,
        texts,
        np.array([0, 1, 0, 0]),
        ["x", "y"],
        fp32_path=fp32_file,
        int8_path=int8_file,
        runs=2,
        batch_sizes=(1, 2),
    )
    assert report.size_ratio == pytest.approx(4.0)
    assert report.agreement == pytest.approx(0.75)
    assert report.fp32_accuracy == 1.0
    assert report.int8_accuracy == 0.75
    assert report.accuracy_delta == pytest.approx(-0.25)
    assert report.num_test_examples == 4
    assert report.notes
    assert set(report.latency_fp32) == {1, 2}
    text = format_report(report)
    assert "size (MB)" in text
    assert "75.00%" in text
    assert "batch 2 p95" in text
    path = write_report(report, tmp_path / "reports")
    assert json.loads(path.read_text())["agreement"] == 0.75


def test_compare_no_notes_for_large_sets(tmp_path: Path) -> None:
    f = tmp_path / "f.onnx"
    f.write_bytes(b"x")
    fp32, _ = _fake_predictors()
    report = compare_models(
        fp32,
        fp32,
        ["a"] * 120,
        np.zeros(120, dtype=int),
        ["x", "y"],
        fp32_path=f,
        int8_path=f,
        runs=1,
        batch_sizes=(1,),
    )
    assert report.notes == []
    assert report.agreement == 1.0
