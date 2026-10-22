"""KEEP/REMOVE filter for lecture sentences, backed by the fine-tuned MiniLM (INT8 ONNX).

Reads one sentence at a time and keeps those whose probability of being teaching content is at
least ``threshold``. The threshold sits below 0.5 on purpose: losing real content is worse than
keeping a little junk. Needs ``onnxruntime`` and ``tokenizers`` (loaded lazily); torch is not used.
"""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_DIR_ENV = "EDUSYNC_C3_MODEL_DIR"
DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2] / "blueprint_model" / "artifacts" / "quantized"
)
DEFAULT_THRESHOLD = 0.4
KEEP_LABEL = "1"
_BATCH = 32


class ModelUnavailableError(RuntimeError):
    """The model files or the ONNX libraries are missing."""


@dataclass(frozen=True)
class ScoredSentence:
    text: str
    p_keep: float
    keep: bool


def model_dir() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)


class KeepFilter:
    """``session`` needs ``run(None, feeds)``; ``tokenizer`` needs ``encode_batch(texts)``."""

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        labels: list[str],
        max_length: int,
        pad_id: int = 0,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if KEEP_LABEL not in labels:
            raise ValueError(f"Model labels {labels} do not contain the KEEP label {KEEP_LABEL!r}")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be between 0 and 1 (exclusive)")
        self.session, self.tokenizer = session, tokenizer
        self.keep_index = labels.index(KEEP_LABEL)
        self.max_length, self.pad_id, self.threshold = max_length, pad_id, threshold

    @classmethod
    def from_dir(
        cls, path: Path | None = None, threshold: float = DEFAULT_THRESHOLD
    ) -> "KeepFilter":
        directory = Path(path) if path else model_dir()
        models = sorted(directory.glob("model_*.onnx"))
        needed = [directory / "tokenizer.json", directory / "inference_config.json"]
        if len(models) != 1 or not all(p.is_file() for p in needed):
            raise ModelUnavailableError(
                f"C3 model not found in {directory} (need one model_*.onnx, tokenizer.json and "
                f"inference_config.json). Pull the Git LFS files or set {MODEL_DIR_ENV}."
            )
        try:
            import onnxruntime
            import tokenizers
        except ImportError as exc:
            raise ModelUnavailableError(
                "onnxruntime and tokenizers are required for the C3 filter: "
                "uv sync --package edusync-blueprint-model --extra onnx"
            ) from exc
        info = json.loads(needed[1].read_text(encoding="utf-8"))
        tok = tokenizers.Tokenizer.from_file(str(needed[0]))
        tok.no_padding()
        tok.enable_truncation(max_length=int(info["max_length"]))
        session = onnxruntime.InferenceSession(str(models[0]), providers=["CPUExecutionProvider"])
        return cls(
            session, tok, list(info["labels"]), int(info["max_length"]),
            tok.token_to_id("[PAD]") or 0, threshold,
        )  # fmt: skip

    def _logits(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        encodings = self.tokenizer.encode_batch(texts)
        ids = [list(e.ids)[: self.max_length] for e in encodings]
        masks = [list(e.attention_mask)[: self.max_length] for e in encodings]
        width = max(len(i) for i in ids)
        input_ids = np.full((len(ids), width), self.pad_id, dtype=np.int64)
        attention = np.zeros((len(ids), width), dtype=np.int64)
        for row, (i, m) in enumerate(zip(ids, masks, strict=True)):
            input_ids[row, : len(i)] = i
            attention[row, : len(m)] = m
        out = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention})
        return np.asarray(out[0], dtype=np.float64).tolist()

    def p_keep(self, sentences: list[str]) -> list[float]:
        """Probability that each sentence is teaching content."""
        scores: list[float] = []
        for start in range(0, len(sentences), _BATCH):
            for row in self._logits(sentences[start : start + _BATCH]):
                top = max(row)
                exp = [math.exp(v - top) for v in row]
                scores.append(exp[self.keep_index] / sum(exp))
        return scores

    def score(self, sentences: list[str]) -> list[ScoredSentence]:
        sentences = [s for s in (t.strip() for t in sentences) if s]
        return [
            ScoredSentence(text, p, p >= self.threshold)
            for text, p in zip(sentences, self.p_keep(sentences), strict=True)
        ]

    def keep_only(self, sentences: list[str]) -> list[str]:
        return [s.text for s in self.score(sentences) if s.keep]
