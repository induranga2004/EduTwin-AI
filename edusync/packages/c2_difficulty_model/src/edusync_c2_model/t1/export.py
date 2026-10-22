"""Torch-free parts of the ONNX export: the I/O contract, parity check, sidecar files and the
ONNX-side predictor. ``torch_impl.export_onnx`` does the actual ``torch.onnx.export``."""

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from edusync_c2_model.t1.config import BASE_MODEL, BASE_REVISION, T1Config
from edusync_c2_model.t1.data import Encode, pad_batch

OPSET = 17
INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "score"
FP32_NAME = "model_fp32.onnx"
INT8_NAME = "model_int8.onnx"
PARITY_ATOL = 1e-3
PARITY_TEXTS = (
    "A short sentence.",
    "A somewhat longer sentence to check that padding and the attention mask behave properly.",
    "Another example, quite different in length from the first two, to vary the batch.",
)
LICENCE_NOTE = (
    "Fine-tuned on CLEAR (CC BY-NC-SA 4.0, non-commercial) and OneStopEnglish (CC BY-SA); the "
    "weights are derived from them, so non-commercial use only unless the CLEAR terms are cleared. "
    "Base model all-MiniLM-L6-v2 is Apache-2.0."
)
TRAINED_ON = (
    "CLEAR train split (reading ease, negated) plus OneStopEnglish train split (levels 0/1/2), "
    "each "
    "z-scored with its own train mean/std. General English, not technical lecture slides."
)

Run = Callable[[dict[str, np.ndarray]], np.ndarray]


class ContractError(ValueError):
    """The ONNX model does not match the input/output contract."""


def check_io_contract(
    inputs: Sequence[tuple[str, str, Sequence]], outputs: Sequence[tuple[str, str, Sequence]]
) -> None:
    """Each entry is ``(name, onnx_type, shape)``. Raises ``ContractError`` on any mismatch."""
    if [n for n, _, _ in inputs] != list(INPUT_NAMES):
        raise ContractError(
            f"input names must be {list(INPUT_NAMES)}, got {[n for n, _, _ in inputs]}"
        )
    for name, kind, shape in inputs:
        if kind != "tensor(int64)" or len(shape) != 2:
            raise ContractError(
                f"input {name} must be int64 [batch, seq], got {kind} {list(shape)}"
            )
        if not all(isinstance(d, str) or d is None for d in shape):
            raise ContractError(f"input {name} must have dynamic batch and sequence axes")
    if len(outputs) != 1 or outputs[0][0] != OUTPUT_NAME:
        raise ContractError(f"output must be the single tensor {OUTPUT_NAME!r}")
    _, kind, shape = outputs[0]
    if kind != "tensor(float)" or len(shape) != 2 or shape[1] != 1:
        raise ContractError(f"output must be float32 [batch, 1], got {kind} {list(shape)}")


def check_parity(expected: np.ndarray, actual: np.ndarray, atol: float = PARITY_ATOL) -> float:
    """Max absolute difference; ``ValueError`` if shapes differ or it exceeds ``atol``."""
    a, b = np.asarray(expected), np.asarray(actual)
    if a.shape != b.shape:
        raise ValueError(f"parity failed: shapes differ {a.shape} vs {b.shape}")
    diff = float(np.max(np.abs(a - b)))
    if diff > atol:
        raise ValueError(f"ONNX output differs from torch by {diff:.2e} (allowed {atol:.0e})")
    return diff


def make_predictor(
    encode: Encode, run: Run, batch_size: int = 16
) -> Callable[[list[str]], np.ndarray]:
    """Texts -> float64 scores ``[n]`` through an ONNX ``run(feeds) -> [batch, 1]`` callable."""

    def predict(texts: list[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float64)
        ids = encode(list(texts))
        for start in range(0, len(texts), batch_size):
            batch = pad_batch(ids[start : start + batch_size])
            feeds = {INPUT_NAMES[0]: batch.input_ids, INPUT_NAMES[1]: batch.attention_mask}
            out[start : start + len(batch.index)] = run(feeds).reshape(-1)
        return out

    return predict


def parity_check(
    torch_predict: Callable[[list[str]], np.ndarray],
    onnx_predict: Callable[[list[str]], np.ndarray],
    texts: Sequence[str] = PARITY_TEXTS,
    atol: float = PARITY_ATOL,
) -> float:
    """Max abs difference over the whole list AND each text alone (batch of 1)."""
    worst = check_parity(torch_predict(list(texts)), onnx_predict(list(texts)), atol)
    for text in texts:
        worst = max(worst, check_parity(torch_predict([text]), onnx_predict([text]), atol))
    return worst


def inference_config(config: T1Config, scalers: dict[str, dict]) -> dict:
    return {
        "max_length": config.max_length,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "target": "difficulty_z",
        "output": OUTPUT_NAME,
        "inputs": list(INPUT_NAMES),
        "pooling": "mean over attention mask, inside the graph",
        "trained_on": TRAINED_ON,
        "licence": LICENCE_NOTE,
        "train_mean_std": {k: {"mean": v["mean"], "std": v["std"]} for k, v in scalers.items()},
    }


def write_sidecars(
    out_dir: Path, tokenizer_json: Path, config: T1Config, scalers: dict[str, dict]
) -> None:
    """Copy tokenizer.json and write inference_config.json next to the models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_json, out_dir / "tokenizer.json")
    text = json.dumps(inference_config(config, scalers), indent=2, sort_keys=True)
    (out_dir / "inference_config.json").write_text(text + "\n", encoding="utf-8")
