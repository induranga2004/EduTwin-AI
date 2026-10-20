"""Post-training dynamic INT8 quantization and the fp32 vs int8 comparison."""

import json
import logging
import shutil
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from edusync_blueprint_model.config import BlueprintConfig
from edusync_blueprint_model.metrics import accuracy, macro_f1
from edusync_blueprint_model.onnx_export import FP32_NAME, SIDECARS

logger = logging.getLogger(__name__)

INT8_NAME = "model_int8.onnx"
BATCH_SIZES = (1, 16)


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


class LatencyStats(BaseModel):
    p50_ms: float
    p95_ms: float
    mean_ms: float


def benchmark(
    fn: Callable[[], object],
    runs: int,
    timer: Callable[[], float] = time.perf_counter,
    warmup: int = 3,
) -> LatencyStats:
    """Time ``fn`` ``runs`` times after ``warmup`` untimed calls; results in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = timer()
        fn()
        samples.append((timer() - start) * 1000.0)
    return LatencyStats(
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        mean_ms=statistics.fmean(samples),
    )


class ComparisonReport(BaseModel):
    fp32_size_mb: float
    int8_size_mb: float
    size_ratio: float
    latency_fp32: dict[int, LatencyStats]
    latency_int8: dict[int, LatencyStats]
    fp32_accuracy: float
    int8_accuracy: float
    accuracy_delta: float
    fp32_macro_f1: float
    int8_macro_f1: float
    macro_f1_delta: float
    agreement: float
    num_test_examples: int
    notes: list[str] = []


Predict = Callable[[list[str]], np.ndarray]


def compare_models(
    fp32_predict: Predict,
    int8_predict: Predict,
    texts: list[str],
    y_true: np.ndarray,
    labels: list[str],
    *,
    fp32_path: Path,
    int8_path: Path,
    runs: int = 50,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
    timer: Callable[[], float] = time.perf_counter,
) -> ComparisonReport:
    """Compare size, latency and quality; predictors map texts to ``(n, classes)`` logits."""
    pred32 = np.argmax(fp32_predict(texts), axis=-1)
    pred8 = np.argmax(int8_predict(texts), axis=-1)
    n = len(labels)
    latency: dict[str, dict[int, LatencyStats]] = {"fp32": {}, "int8": {}}
    for size in batch_sizes:
        batch = [texts[i % len(texts)] for i in range(size)]
        latency["fp32"][size] = benchmark(lambda b=batch: fp32_predict(b), runs, timer)
        latency["int8"][size] = benchmark(lambda b=batch: int8_predict(b), runs, timer)
    size32, size8 = file_size_mb(fp32_path), file_size_mb(int8_path)
    acc32, acc8 = accuracy(y_true, pred32), accuracy(y_true, pred8)
    f1_32, f1_8 = macro_f1(y_true, pred32, n), macro_f1(y_true, pred8, n)
    notes = []
    if len(texts) < 100:
        notes.append(f"Only {len(texts)} test examples: accuracy differences are not significant.")
    return ComparisonReport(
        fp32_size_mb=size32,
        int8_size_mb=size8,
        size_ratio=size32 / size8 if size8 else 0.0,
        latency_fp32=latency["fp32"],
        latency_int8=latency["int8"],
        fp32_accuracy=acc32,
        int8_accuracy=acc8,
        accuracy_delta=acc8 - acc32,
        fp32_macro_f1=f1_32,
        int8_macro_f1=f1_8,
        macro_f1_delta=f1_8 - f1_32,
        agreement=float(np.mean(pred32 == pred8)) if len(texts) else 0.0,
        num_test_examples=len(texts),
        notes=notes,
    )


def format_report(report: ComparisonReport) -> str:
    """Aligned plain-text table for the terminal."""
    rows = [
        ("metric", "fp32", "int8", "change"),
        (
            "size (MB)",
            f"{report.fp32_size_mb:.1f}",
            f"{report.int8_size_mb:.1f}",
            f"{report.size_ratio:.2f}x smaller",
        ),
    ]
    for size in sorted(report.latency_fp32):
        a, b = report.latency_fp32[size], report.latency_int8[size]
        for name in ("p50_ms", "p95_ms"):
            va, vb = getattr(a, name), getattr(b, name)
            speed = f"{va / vb:.2f}x faster" if vb else "n/a"
            rows.append((f"batch {size} {name[:3]} (ms)", f"{va:.2f}", f"{vb:.2f}", speed))
    rows.append(
        (
            "accuracy",
            f"{report.fp32_accuracy:.4f}",
            f"{report.int8_accuracy:.4f}",
            f"{report.accuracy_delta:+.4f}",
        )
    )
    rows.append(
        (
            "macro F1",
            f"{report.fp32_macro_f1:.4f}",
            f"{report.int8_macro_f1:.4f}",
            f"{report.macro_f1_delta:+.4f}",
        )
    )
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append(
        f"prediction agreement: {report.agreement:.2%} on {report.num_test_examples} texts"
    )
    lines.extend(f"note: {note}" for note in report.notes)
    return "\n".join(lines)


def quantize_model(  # pragma: no cover - needs onnxruntime
    config: BlueprintConfig, *, fp32_path: Path | None = None, int8_path: Path | None = None
) -> Path:
    """Dynamic INT8 quantization of the exported ONNX model, with sidecar files copied over."""
    from edusync_blueprint_model.deps import require

    require("onnx", "onnx")
    quant = require("onnxruntime.quantization", "onnx")
    fp32 = Path(fp32_path) if fp32_path else config.artifacts.onnx_dir / FP32_NAME
    if not fp32.is_file():
        raise FileNotFoundError(f"{fp32} not found; run the export step first")
    out = Path(int8_path) if int8_path else config.artifacts.quantized_dir / INT8_NAME
    out.parent.mkdir(parents=True, exist_ok=True)

    source = fp32
    prepped = out.parent / "model_prep.onnx"
    # Pre-processing (ONNX shape inference + graph optimisation) is recommended before quantizing.
    # Symbolic shape inference is skipped on purpose: it always fails on this model's dynamic
    # sequence axis ("Incomplete symbolic shape inference") and each failed attempt leaves a
    # model-sized temporary *.data file in the working directory. Dynamic quantization does
    # not need it.
    from onnxruntime.quantization.shape_inference import quant_pre_process

    try:
        quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
        source = prepped
    except Exception as exc:
        logger.warning("Pre-processing skipped (%s); quantizing the raw exported model", exc)

    qc = config.quantize
    quant.quantize_dynamic(
        str(source),
        str(out),
        weight_type=quant.QuantType.QInt8 if qc.weight_type == "qint8" else quant.QuantType.QUInt8,
        per_channel=qc.per_channel,
        op_types_to_quantize=qc.op_types,
    )
    prepped.unlink(missing_ok=True)
    for name in SIDECARS:
        shutil.copyfile(fp32.parent / name, out.parent / name)
    return out


def write_report(report: ComparisonReport, reports_dir: Path) -> Path:
    """Save ``quantization_report.json`` and return its path."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "quantization_report.json"
    path.write_text(json.dumps(json.loads(report.model_dump_json()), indent=2), encoding="utf-8")
    return path
