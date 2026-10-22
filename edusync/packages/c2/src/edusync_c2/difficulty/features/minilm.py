"""Pure-numpy MiniLM (BERT) sentence encoder over local safetensors weights.

Reproduces the sentence-transformers pipeline of all-MiniLM-L6-v2: WordPiece tokens wrapped in
[CLS]/[SEP], post-LayerNorm BERT layers with absolute positions, mean pooling over all tokens and
L2 normalisation. numpy is imported lazily inside functions, so `import edusync_c2` never needs it.

Determinism: every text is encoded on its own (no batching, no padding, so no attention mask is
needed and a text gets the same vector alone or inside any batch). All arrays are float32. The
exact (erf) GELU of the model is used with a numpy-only erf approximation (max error 1.5e-7).
"""

import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edusync_c2.difficulty.features.wordpiece import WordPieceTokenizer

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[Any]

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
DEFAULT_MAX_LENGTH = 128
_LFS_MAGIC = b"version https://git-lfs"
_LFS_MIN_BYTES = 1024
_ERF_P = 0.3275911
_ERF_COEFFS = (1.061405429, -1.453152027, 1.421413741, -0.284496736, 0.254829592)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_CLS, _SEP = "[CLS]", "[SEP]"


class SemanticModelError(Exception):
    """The semantic model cannot be used; `reason` is a short machine-readable cause."""

    def __init__(self, reason: str, message: str | None = None) -> None:
        super().__init__(message or reason)
        self.reason = reason


def is_lfs_pointer(path: Path) -> bool:
    """True for a Git LFS pointer stub (or any file too small to hold real weights)."""
    if path.stat().st_size < _LFS_MIN_BYTES:
        return True
    with path.open("rb") as handle:
        return handle.read(len(_LFS_MAGIC)) == _LFS_MAGIC


def _corrupt(message: str) -> SemanticModelError:
    return SemanticModelError("corrupt", message)


def _tensor_entry(name: str, entry: Any) -> tuple[str, tuple[int, ...], int, int]:
    try:
        dtype = entry["dtype"]
        shape = tuple(int(dim) for dim in entry["shape"])
        begin, end = (int(v) for v in entry["data_offsets"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _corrupt(f"bad safetensors entry {name!r}") from exc
    if begin < 0 or end < begin or any(dim < 0 for dim in shape):
        raise _corrupt(f"bad safetensors extent for {name!r}")
    return dtype, shape, begin, end


def read_safetensors(path: Path, names: Collection[str] | None = None) -> dict[str, "Array"]:
    """Read float32 tensors from a safetensors file (all, or only `names`).

    Layout: 8-byte little-endian header length, a JSON header, then the raw data. Only the
    requested tensors are validated and read, so unrelated entries (such as int64 position ids)
    may use other dtypes. Raises SemanticModelError("corrupt") for any malformed or truncated file;
    OSError propagates for unreadable files.
    """
    import numpy as np

    with path.open("rb") as handle:
        total = path.stat().st_size
        prefix = handle.read(8)
        if len(prefix) < 8:
            raise _corrupt("safetensors file too short")
        header_len = int.from_bytes(prefix, "little")
        if header_len > total - 8:
            raise _corrupt("safetensors header exceeds file")
        try:
            header = json.loads(handle.read(header_len))
        except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
            raise _corrupt("safetensors header is not JSON") from exc
        if not isinstance(header, dict):
            raise _corrupt("safetensors header is not an object")
        data_start = 8 + header_len
        data_size = total - data_start
        tensors: dict[str, Array] = {}
        for name, entry in header.items():
            if name == "__metadata__" or (names is not None and name not in names):
                continue
            dtype, shape, begin, end = _tensor_entry(name, entry)
            if dtype != "F32":
                raise _corrupt(f"unsupported dtype {dtype!r} for {name!r}")
            count = math.prod(shape)
            if end - begin != count * 4:
                raise _corrupt(f"size mismatch for {name!r}")
            if end > data_size:
                raise _corrupt("safetensors file is truncated")
            handle.seek(data_start + begin)
            flat = np.fromfile(handle, dtype="<f4", count=count)
            tensors[name] = flat.astype(np.float32, copy=False).reshape(shape)
        return tensors


def erf_approx(x: "Array") -> "Array":
    """Vectorised erf (Abramowitz-Stegun 7.1.26), absolute error below 1.5e-7, dtype kept."""
    import numpy as np

    magnitude = np.abs(x)
    t = 1.0 / (1.0 + _ERF_P * magnitude)
    poly = _ERF_COEFFS[0]
    for coeff in _ERF_COEFFS[1:]:
        poly = poly * t + coeff
    value = 1.0 - poly * t * np.exp(-magnitude * magnitude)
    return np.where(x < 0, -value, value).astype(x.dtype, copy=False)


def gelu(x: "Array") -> "Array":
    """Exact GELU, x * Phi(x), using erf_approx."""
    return 0.5 * x * (1.0 + erf_approx(x * _INV_SQRT2))


def _layer_norm(x: "Array", weight: "Array", bias: "Array", eps: float) -> "Array":
    import numpy as np

    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = (centered * centered).mean(axis=-1, keepdims=True)
    return centered / np.sqrt(var + eps) * weight + bias


@dataclass(frozen=True)
class MiniLMConfig:
    """BERT dimensions read from the model's config.json."""

    hidden_size: int = 384
    num_layers: int = 6
    num_heads: int = 12
    intermediate_size: int = 1536
    layer_norm_eps: float = 1e-12
    max_positions: int = 512
    hidden_act: str = "gelu"

    def __post_init__(self) -> None:
        dims = (self.hidden_size, self.num_layers, self.num_heads, self.intermediate_size)
        if min(dims) < 1 or self.max_positions < 3 or self.hidden_size % self.num_heads:
            raise SemanticModelError("incompatible", "invalid model dimensions")
        if self.hidden_act != "gelu":
            raise SemanticModelError("incompatible", "only the exact gelu activation is supported")

    @classmethod
    def from_file(cls, path: Path) -> "MiniLMConfig":
        """Parse a Hugging Face BERT config.json."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SemanticModelError("missing", "model config not found") from exc
        except ValueError as exc:
            raise _corrupt("model config is not JSON") from exc
        try:
            return cls(
                hidden_size=int(raw["hidden_size"]),
                num_layers=int(raw["num_hidden_layers"]),
                num_heads=int(raw["num_attention_heads"]),
                intermediate_size=int(raw["intermediate_size"]),
                layer_norm_eps=float(raw["layer_norm_eps"]),
                max_positions=int(raw["max_position_embeddings"]),
                hidden_act=str(raw["hidden_act"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _corrupt("model config has missing or invalid fields") from exc


def required_tensors(config: MiniLMConfig) -> dict[str, tuple[int | None, ...]]:
    """Tensor names the encoder reads with their shapes (None = free first dimension)."""
    h, i = config.hidden_size, config.intermediate_size
    shapes: dict[str, tuple[int | None, ...]] = {
        "embeddings.word_embeddings.weight": (None, h),
        "embeddings.position_embeddings.weight": (None, h),
        "embeddings.token_type_embeddings.weight": (None, h),
        "embeddings.LayerNorm.weight": (h,),
        "embeddings.LayerNorm.bias": (h,),
    }
    for n in range(config.num_layers):
        base = f"encoder.layer.{n}."
        for part in ("query", "key", "value"):
            shapes[f"{base}attention.self.{part}.weight"] = (h, h)
            shapes[f"{base}attention.self.{part}.bias"] = (h,)
        shapes[f"{base}attention.output.dense.weight"] = (h, h)
        shapes[f"{base}attention.output.dense.bias"] = (h,)
        shapes[f"{base}attention.output.LayerNorm.weight"] = (h,)
        shapes[f"{base}attention.output.LayerNorm.bias"] = (h,)
        shapes[f"{base}intermediate.dense.weight"] = (i, h)
        shapes[f"{base}intermediate.dense.bias"] = (i,)
        shapes[f"{base}output.dense.weight"] = (h, i)
        shapes[f"{base}output.dense.bias"] = (h,)
        shapes[f"{base}output.LayerNorm.weight"] = (h,)
        shapes[f"{base}output.LayerNorm.bias"] = (h,)
    return shapes


def _shape_matches(actual: tuple[int, ...], expected: tuple[int | None, ...]) -> bool:
    return len(actual) == len(expected) and all(
        want is None or got == want for got, want in zip(actual, expected, strict=True)
    )


@dataclass(frozen=True)
class _Layer:
    w_qkv: "Array"
    b_qkv: "Array"
    w_attn_out: "Array"
    b_attn_out: "Array"
    ln1_w: "Array"
    ln1_b: "Array"
    w_inter: "Array"
    b_inter: "Array"
    w_out: "Array"
    b_out: "Array"
    ln2_w: "Array"
    ln2_b: "Array"


class MiniLMEncoder:
    """Sentence encoder; immutable after construction and safe to share between threads."""

    def __init__(
        self,
        weights: Mapping[str, "Array"],
        config: MiniLMConfig,
        tokenizer: WordPieceTokenizer,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
    ) -> None:
        import numpy as np

        for name, expected in required_tensors(config).items():
            if name not in weights:
                raise SemanticModelError("incompatible", f"missing tensor {name}")
            if not _shape_matches(tuple(weights[name].shape), expected):
                raise SemanticModelError("incompatible", f"unexpected shape for {name}")
        word = weights["embeddings.word_embeddings.weight"]
        pos = weights["embeddings.position_embeddings.weight"]
        if word.shape[0] < tokenizer.vocab_size:
            raise SemanticModelError("incompatible", "weights have fewer rows than the vocabulary")
        if not 3 <= max_length <= min(config.max_positions, pos.shape[0]):
            raise SemanticModelError("incompatible", "max_length outside the position table")
        try:
            self._cls, self._sep = tokenizer.ids([_CLS, _SEP])
        except KeyError as exc:
            raise SemanticModelError("incompatible", "vocabulary lacks [CLS]/[SEP]") from exc
        self.config = config
        self.max_length = max_length
        self.sha: str | None = None  # weights fingerprint, set by the loader
        self._tokenizer = tokenizer
        self._word = np.asarray(word, dtype=np.float32)
        self._pos = np.asarray(pos, dtype=np.float32)
        self._type0 = np.asarray(weights["embeddings.token_type_embeddings.weight"][0], np.float32)
        self._ln_w = weights["embeddings.LayerNorm.weight"]
        self._ln_b = weights["embeddings.LayerNorm.bias"]
        self._layers = tuple(self._build_layer(weights, n) for n in range(config.num_layers))

    @staticmethod
    def _build_layer(weights: Mapping[str, "Array"], n: int) -> _Layer:
        import numpy as np

        base = f"encoder.layer.{n}."
        qkv_w = [weights[f"{base}attention.self.{p}.weight"].T for p in ("query", "key", "value")]
        qkv_b = [weights[f"{base}attention.self.{p}.bias"] for p in ("query", "key", "value")]
        return _Layer(
            w_qkv=np.ascontiguousarray(np.concatenate(qkv_w, axis=1), dtype=np.float32),
            b_qkv=np.concatenate(qkv_b).astype(np.float32),
            w_attn_out=np.ascontiguousarray(weights[f"{base}attention.output.dense.weight"].T),
            b_attn_out=weights[f"{base}attention.output.dense.bias"],
            ln1_w=weights[f"{base}attention.output.LayerNorm.weight"],
            ln1_b=weights[f"{base}attention.output.LayerNorm.bias"],
            w_inter=np.ascontiguousarray(weights[f"{base}intermediate.dense.weight"].T),
            b_inter=weights[f"{base}intermediate.dense.bias"],
            w_out=np.ascontiguousarray(weights[f"{base}output.dense.weight"].T),
            b_out=weights[f"{base}output.dense.bias"],
            ln2_w=weights[f"{base}output.LayerNorm.weight"],
            ln2_b=weights[f"{base}output.LayerNorm.bias"],
        )

    @classmethod
    def from_directory(
        cls, model_dir: Path, *, max_length: int = DEFAULT_MAX_LENGTH
    ) -> "MiniLMEncoder":
        """Load model.safetensors and config.json from `model_dir` plus the vendored vocabulary.

        The vendored bert_uncased_vocab.txt is byte-identical to the model's own vocab.txt.
        """
        weights_path = model_dir / WEIGHTS_FILE
        if not weights_path.is_file():
            raise SemanticModelError("missing", "model weights not found")
        if is_lfs_pointer(weights_path):
            raise SemanticModelError("lfs_pointer", "model weights are a Git LFS pointer")
        config = MiniLMConfig.from_file(model_dir / CONFIG_FILE)
        weights = read_safetensors(weights_path, required_tensors(config))
        return cls(weights, config, WordPieceTokenizer.from_resources(), max_length=max_length)

    @property
    def dims(self) -> int:
        """Embedding dimensionality."""
        return self.config.hidden_size

    def encode(self, texts: Sequence[str]) -> "Array":
        """Unit-norm float32 embeddings, shape (len(texts), dims); each text is encoded alone."""
        import numpy as np

        out = np.empty((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row] = self._encode_one(text)
        return out

    def _encode_one(self, text: str) -> "Array":
        import numpy as np

        pieces = self._tokenizer.tokenize(text)[: self.max_length - 2]
        ids = [self._cls, *self._tokenizer.ids(pieces), self._sep]
        x = self._word[ids] + self._pos[: len(ids)] + self._type0
        x = _layer_norm(x, self._ln_w, self._ln_b, self.config.layer_norm_eps)
        for layer in self._layers:
            x = self._block(x, layer)
        pooled = x.mean(axis=0)
        norm = max(float(np.sqrt(np.sum(pooled * pooled))), 1e-12)
        return (pooled / norm).astype(np.float32)

    def _block(self, x: "Array", layer: _Layer) -> "Array":
        import numpy as np

        cfg = self.config
        length, hidden = x.shape
        head_dim = hidden // cfg.num_heads
        qkv = x @ layer.w_qkv + layer.b_qkv
        q, k, v = (
            qkv[:, n * hidden : (n + 1) * hidden]
            .reshape(length, cfg.num_heads, head_dim)
            .transpose(1, 0, 2)
            for n in range(3)
        )
        scores = (q @ k.transpose(0, 2, 1)) * (1.0 / math.sqrt(head_dim))
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        context = (probs @ v).transpose(1, 0, 2).reshape(length, hidden)
        x = _layer_norm(
            x + context @ layer.w_attn_out + layer.b_attn_out,
            layer.ln1_w,
            layer.ln1_b,
            cfg.layer_norm_eps,
        )
        inner = gelu(x @ layer.w_inter + layer.b_inter)
        return _layer_norm(
            x + inner @ layer.w_out + layer.b_out, layer.ln2_w, layer.ln2_b, cfg.layer_norm_eps
        )
