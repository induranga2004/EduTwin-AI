"""The thin torch / onnxruntime layer of T1. Imported only inside the training container.

Every torch, transformers, tokenizers, onnx and onnxruntime import is lazy, and the file is
exercised by the ``ml``-marked tests (``tests/test_t1_ml.py``), which CI deselects. It is the one
file in the package that is omitted from the CI coverage report (see ``pyproject.toml``); all
decisions and numerics around it live in torch-free modules that are tested with fakes.
"""

import copy
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from edusync_c2_model.t1.config import T1Config
from edusync_c2_model.t1.data import Batch, Encode
from edusync_c2_model.t1.export import INPUT_NAMES, OPSET, OUTPUT_NAME, PARITY_TEXTS

logger = logging.getLogger(__name__)
_CLASSES: dict[str, Any] = {}


def versions() -> dict[str, str]:
    import onnx
    import onnxruntime
    import tokenizers
    import torch
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
    }


def set_threads(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def seed_everything(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)


def load_encoder(tokenizer_json: Path, max_length: int) -> Encode:
    """Encode with the base ``tokenizer.json``: truncate to ``max_length``, no padding."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=max_length)

    def encode(texts: list[str]) -> list[list[int]]:
        return [e.ids for e in tokenizer.encode_batch(texts)]

    return encode


def _model_class() -> Any:
    if "cls" in _CLASSES:
        return _CLASSES["cls"]
    import torch

    class MiniLMRegressor(torch.nn.Module):
        """Encoder -> mean pooling over real tokens -> dropout -> Linear(hidden, 1)."""

        def __init__(self, encoder: Any, dropout: float) -> None:
            super().__init__()
            self.encoder = encoder
            self.dropout = torch.nn.Dropout(dropout)
            self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
            pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            return self.head(self.dropout(pooled))

    _CLASSES["cls"] = MiniLMRegressor
    return MiniLMRegressor


def build_model(base_dir: Path, dropout: float) -> Any:
    """Load the base MiniLM (eager attention, no pooler) and attach a fresh regression head."""
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(
        str(base_dir), add_pooling_layer=False, attn_implementation="eager"
    )
    return _model_class()(encoder, dropout)


def _tensors(batch: Batch) -> tuple[Any, Any]:
    import torch

    return torch.from_numpy(batch.input_ids), torch.from_numpy(batch.attention_mask)


class TorchBackend:
    """Implements ``trainer.Backend`` with AdamW (4 parameter groups) and Huber/MSE loss."""

    def __init__(self, model: Any, config: T1Config) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.config = config
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            frozen = config.freeze_encoder and name.startswith("encoder.")
            param.requires_grad = not frozen
            if frozen:
                continue
            bucket = no_decay if name.endswith(".bias") or "LayerNorm" in name else decay
            bucket.append((name, param))
        groups = []
        for items, wd in ((decay, config.weight_decay), (no_decay, 0.0)):
            for is_head, lr in ((False, config.lr_encoder), (True, config.lr_head)):
                params = [p for n, p in items if n.startswith("head.") == is_head]
                if params:
                    groups.append({"params": params, "lr": lr, "base_lr": lr, "weight_decay": wd})
        self.optimizer = torch.optim.AdamW(groups)
        self.loss_fn = (
            torch.nn.HuberLoss(delta=config.huber_delta)
            if config.loss == "huber"
            else torch.nn.MSELoss()
        )

    def train_step(self, batch: Batch, lr_scale: float) -> float:
        self.model.train()
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * lr_scale
        ids, mask = _tensors(batch)
        out = self.model(ids, mask).squeeze(-1)
        loss = self.loss_fn(out, self.torch.from_numpy(batch.targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.item())

    def predict_batch(self, batch: Batch) -> np.ndarray:
        self.model.eval()
        ids, mask = _tensors(batch)
        with self.torch.no_grad():
            return self.model(ids, mask).squeeze(-1).numpy().astype(np.float64)

    def snapshot(self) -> Any:
        return copy.deepcopy(self.model.state_dict())

    def restore(self, state: Any) -> None:
        self.model.load_state_dict(state)


def torch_predictor(
    model: Any, encode: Encode, batch_size: int = 16
) -> Callable[[list[str]], np.ndarray]:
    """Texts -> scores with the (eval-mode) torch model."""
    import torch

    from edusync_c2_model.t1.data import pad_batch

    def predict(texts: list[str]) -> np.ndarray:
        model.eval()
        ids = encode(list(texts))
        out = np.zeros(len(texts), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = pad_batch(ids[start : start + batch_size])
                a, b = _tensors(batch)
                out[start : start + len(batch.index)] = model(a, b).reshape(-1).numpy()
        return out

    return predict


def export_onnx(model: Any, encode: Encode, path: Path) -> str:
    """Write the fp32 ONNX graph (opset 17, dynamic batch/sequence). Returns the exporter used."""
    import torch

    model.eval()
    batch = _example_batch(encode)
    kwargs: dict[str, Any] = {
        "input_names": list(INPUT_NAMES),
        "output_names": [OUTPUT_NAME],
        "dynamic_axes": {
            INPUT_NAMES[0]: {0: "batch", 1: "sequence"},
            INPUT_NAMES[1]: {0: "batch", 1: "sequence"},
            OUTPUT_NAME: {0: "batch"},
        },
        "opset_version": OPSET,
    }
    with torch.no_grad():
        try:
            torch.onnx.export(model, batch, str(path), dynamo=False, **kwargs)
            return "legacy"
        except Exception as exc:
            logger.warning("legacy ONNX export failed (%s); trying the dynamo exporter", exc)
        torch.onnx.export(model, batch, str(path), dynamo=True, **kwargs)
    return "dynamo"


def _example_batch(encode: Encode) -> tuple[Any, Any]:
    from edusync_c2_model.t1.data import pad_batch

    return _tensors(pad_batch(encode(list(PARITY_TEXTS))))


def open_session(path: Path, threads: int) -> tuple[Callable[[dict], np.ndarray], list, list]:
    """``(run, inputs, outputs)`` for an ONNX file; metadata is shaped for ``check_io_contract``."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def run(feeds: dict) -> np.ndarray:
        return session.run(None, feeds)[0]

    meta = lambda items: [(i.name, i.type, i.shape) for i in items]  # noqa: E731
    return run, meta(session.get_inputs()), meta(session.get_outputs())


def quantize_int8(fp32: Path, int8: Path, workdir: Path) -> list[str]:
    """Dynamic INT8 (MatMul only). Pre-processes with ``skip_symbolic_shape=True`` (symbolic shape
    inference leaks model-sized temp files) and deletes any temp file left behind."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    from edusync_c2_model.t1.quantize import remove_new_files, snapshot_dir

    workdir.mkdir(parents=True, exist_ok=True)
    before = snapshot_dir(workdir)
    prepped = workdir / "model_prep.onnx"
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
    quantize_dynamic(
        str(prepped),
        str(int8),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    return remove_new_files(workdir, before, keep=[int8, fp32])
