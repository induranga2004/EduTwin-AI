"""Torch-free parts of INT8 quantization: temp-file cleanup, fp32 vs INT8 comparison and the ship
decision. ``torch_impl.quantize_int8`` calls onnxruntime."""

import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from edusync_c2_model.regression_metrics import evaluate_regressor, rmse, spearman

BATCH_SIZES = (1, 16)
Predict = Callable[[list[str]], np.ndarray]


def snapshot_dir(folder: Path) -> set[Path]:
    return set(Path(folder).iterdir())


def remove_new_files(folder: Path, before: set[Path], keep: Sequence[Path] = ()) -> list[str]:
    """Delete files that appeared in ``folder`` since ``before`` (onnxruntime's symbolic shape
    inference can leak model-sized temp files), except those in ``keep``. Returns the names."""
    kept = {Path(k).resolve() for k in keep}
    removed = []
    for path in sorted(set(Path(folder).iterdir()) - before):
        if path.is_file() and path.resolve() not in kept:
            path.unlink()
            removed.append(path.name)
    return removed


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def latency_ms_per_text(
    predict: Predict,
    texts: Sequence[str],
    batch_size: int,
    runs: int,
    timer: Callable[[], float] = time.perf_counter,
) -> float:
    """Mean wall time per text (ms) for batches of ``batch_size``, after one warm-up call."""
    batch = [texts[i % len(texts)] for i in range(batch_size)]
    predict(batch)
    started = timer()
    for _ in range(runs):
        predict(batch)
    return (timer() - started) * 1000.0 / (runs * batch_size)


def decide_ship(drop: float | None, threshold: float) -> dict:
    """Ship INT8 only if the Spearman drop (fp32 minus INT8) is defined and below ``threshold``."""
    if drop is None:
        return {"ship": "fp32", "drop": None, "reason": "Spearman undefined: keeping fp32"}
    if drop < threshold:
        return {
            "ship": "int8",
            "drop": drop,
            "reason": f"Spearman drop {drop:+.4f} is below the {threshold:.2f} threshold",
        }
    return {
        "ship": "fp32",
        "drop": drop,
        "reason": f"Spearman drop {drop:+.4f} reaches the {threshold:.2f} threshold: keeping fp32",
    }


def compare_models(
    fp32_predict: Predict,
    int8_predict: Predict,
    eval_sets: dict[str, tuple[list[str], np.ndarray, list[str]]],
    *,
    fp32_path: Path,
    int8_path: Path,
    threshold: float,
    runs: int = 10,
    batch_sizes: Sequence[int] = BATCH_SIZES,
    timer: Callable[[], float] = time.perf_counter,
) -> dict:
    """Quality per eval set (``name -> (texts, z_targets, group_ids)``), size, latency, decision.

    The decision uses the MEAN Spearman difference across the eval sets.
    """
    quality: dict[str, dict] = {}
    drops = []
    for name, (texts, targets, groups) in eval_sets.items():
        p32, p8 = fp32_predict(texts), int8_predict(texts)
        s32, s8 = spearman(targets, p32), spearman(targets, p8)
        quality[name] = {
            "n": len(texts),
            "fp32": evaluate_regressor(targets, p32, groups, 0, n_boot=200),
            "int8": evaluate_regressor(targets, p8, groups, 0, n_boot=200),
            "max_abs_diff": float(np.max(np.abs(p32 - p8))),
            "rmse_between": rmse(p32, p8),
        }
        if s32 is not None and s8 is not None:
            drops.append(s32 - s8)
    sample = [t for texts, _, _ in eval_sets.values() for t in texts]
    latency = {
        size: {
            "fp32": latency_ms_per_text(fp32_predict, sample, size, runs, timer),
            "int8": latency_ms_per_text(int8_predict, sample, size, runs, timer),
        }
        for size in batch_sizes
    }
    mean_drop = float(np.mean(drops)) if drops else None
    decision = decide_ship(mean_drop, threshold)
    return {
        "quality": quality,
        "latency_ms_per_text": latency,
        "size_mb": {"fp32": file_size_mb(fp32_path), "int8": file_size_mb(int8_path)},
        "decision": decision,
        "threshold": threshold,
    }
