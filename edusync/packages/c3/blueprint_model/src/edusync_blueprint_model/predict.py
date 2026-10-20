"""Torch-free inference with ONNX Runtime (never imports torch or transformers)."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from edusync_blueprint_model.deps import require


class Prediction(BaseModel):
    label: str
    score: float
    probabilities: dict[str, float]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class OnnxClassifier:
    """Tokenizes text, runs an ONNX session and returns labelled predictions.

    ``session`` needs ``run(None, feeds)``; ``tokenizer`` needs ``encode_batch(texts)``
    returning objects with ``ids`` and ``attention_mask`` (a ``tokenizers.Tokenizer``).
    """

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        labels: list[str],
        max_length: int,
        pad_id: int = 0,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.pad_id = pad_id

    def _encode(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        encodings = self.tokenizer.encode_batch(texts)
        ids = [list(e.ids)[: self.max_length] for e in encodings]
        masks = [list(e.attention_mask)[: self.max_length] for e in encodings]
        width = max(len(i) for i in ids)
        input_ids = np.full((len(ids), width), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(ids), width), dtype=np.int64)
        for row, (i, m) in enumerate(zip(ids, masks, strict=True)):
            input_ids[row, : len(i)] = i
            attention[row, : len(m)] = m
        return input_ids, attention

    def logits(self, texts: list[str]) -> np.ndarray:
        """Raw class scores, shape ``(len(texts), num_labels)``."""
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        input_ids, attention = self._encode(texts)
        out = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention})
        return np.asarray(out[0])

    def predict(self, texts: list[str]) -> list[Prediction]:
        probs = softmax(self.logits(texts))
        results = []
        for row in probs:
            best = int(row.argmax())
            results.append(
                Prediction(
                    label=self.labels[best],
                    score=float(row[best]),
                    probabilities={lab: float(p) for lab, p in zip(self.labels, row, strict=True)},
                )
            )
        return results

    @classmethod
    def from_dir(cls, model_dir: Path) -> "OnnxClassifier":  # pragma: no cover - needs ORT
        """Load ``onnx/`` or ``quantized/`` output (one ``model_*.onnx`` plus its sidecars)."""
        model_dir = Path(model_dir)
        models = sorted(model_dir.glob("model_*.onnx"))
        if len(models) != 1 or not (model_dir / "inference_config.json").is_file():
            raise FileNotFoundError(
                f"Expected one model_*.onnx and inference_config.json in {model_dir}; "
                "run the export (and quantize) step first"
            )
        tokenizers = require("tokenizers", "onnx")
        ort = require("onnxruntime", "onnx")
        info = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
        tokenizer = tokenizers.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        tokenizer.no_padding()
        tokenizer.enable_truncation(max_length=int(info["max_length"]))
        pad_id = tokenizer.token_to_id("[PAD]")
        session = ort.InferenceSession(str(models[0]), providers=["CPUExecutionProvider"])
        return cls(session, tokenizer, list(info["labels"]), int(info["max_length"]), pad_id or 0)
