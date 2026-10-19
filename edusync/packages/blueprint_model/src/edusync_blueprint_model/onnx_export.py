"""Export the fine-tuned classifier to ONNX (pooling and head are inside the graph)."""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from edusync_blueprint_model.config import BlueprintConfig

logger = logging.getLogger(__name__)

FP32_NAME = "model_fp32.onnx"
SIDECARS = ("tokenizer.json", "labels.json", "inference_config.json")
PARITY_TEXTS = [
    "A short sentence.",
    "A somewhat longer sentence to check that padding and the attention mask behave.",
    "Another example, quite different in length from the first two, to vary the batch.",
]


def check_parity(torch_logits: np.ndarray, onnx_logits: np.ndarray, atol: float = 1e-3) -> float:
    """Return the max absolute difference; raise ValueError if it exceeds ``atol``."""
    a, b = np.asarray(torch_logits), np.asarray(onnx_logits)
    if a.shape != b.shape:
        raise ValueError(f"Parity check failed: shapes differ {a.shape} vs {b.shape}")
    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if diff > atol:
        raise ValueError(f"ONNX output differs from torch by {diff:.2e} (allowed {atol:.0e})")
    return diff


def write_sidecars(config: BlueprintConfig, finetuned_dir: Path, out_dir: Path) -> None:
    """Copy tokenizer.json and labels.json and write inference_config.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(finetuned_dir / "tokenizer" / "tokenizer.json", out_dir / "tokenizer.json")
    shutil.copyfile(finetuned_dir / "labels.json", out_dir / "labels.json")
    labels = json.loads((finetuned_dir / "labels.json").read_text(encoding="utf-8"))["labels"]
    info = {
        "max_length": config.model.max_length,
        "labels": labels,
        "base_model": config.model.name,
        "base_revision": config.model.revision,
        "source": str(finetuned_dir),
    }
    (out_dir / "inference_config.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def _torch_export(model: Any, args: Any, path: Path) -> str:  # pragma: no cover - needs torch
    """Export with the legacy exporter, falling back to the dynamo exporter."""
    import torch

    kwargs: dict[str, Any] = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        "opset_version": 17,
    }
    try:
        torch.onnx.export(model, args, str(path), dynamo=False, **kwargs)
        return "legacy"
    except Exception as exc:
        logger.warning("Legacy ONNX export failed (%s); trying the dynamo exporter", exc)
    torch.onnx.export(model, args, str(path), dynamo=True, **kwargs)
    return "dynamo"


def export_onnx(  # pragma: no cover - needs torch + onnxruntime
    config: BlueprintConfig, *, finetuned_dir: Path | None = None, out_dir: Path | None = None
) -> Path:
    """Write ``model_fp32.onnx`` (+ sidecars) and verify it matches torch."""
    import torch

    from edusync_blueprint_model.deps import require
    from edusync_blueprint_model.model import load_finetuned

    ort = require("onnxruntime", "onnx")
    transformers = require("transformers", "finetune")
    src = Path(finetuned_dir) if finetuned_dir else config.artifacts.finetuned_dir
    out = Path(out_dir) if out_dir else config.artifacts.onnx_dir
    out.mkdir(parents=True, exist_ok=True)
    model = load_finetuned(src)
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(src / "tokenizer"))
    enc = tokenizer(
        PARITY_TEXTS,
        truncation=True,
        max_length=config.model.max_length,
        padding=True,
        return_tensors="pt",
    )
    args = (enc["input_ids"], enc["attention_mask"])
    path = out / FP32_NAME
    with torch.no_grad():
        exporter = _torch_export(model, args, path)
        expected = model(*args).numpy()
    logger.info("Exported %s using the %s exporter", path, exporter)
    write_sidecars(config, src, out)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {k: enc[k].numpy().astype(np.int64) for k in ("input_ids", "attention_mask")}
    diff = check_parity(expected, session.run(None, feeds)[0])
    # A single short text pads differently, so check dynamic batch/sequence shapes too.
    one = tokenizer(PARITY_TEXTS[0], return_tensors="pt")
    with torch.no_grad():
        expected_one = model(one["input_ids"], one["attention_mask"]).numpy()
    feeds_one = {k: one[k].numpy().astype(np.int64) for k in ("input_ids", "attention_mask")}
    diff = max(diff, check_parity(expected_one, session.run(None, feeds_one)[0]))
    logger.info("ONNX parity max abs diff: %.2e", diff)
    return path
