"""Numpy MiniLM encoder: safetensors reader, erf, a tiny synthetic model, real reference."""

import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edusync_c2.difficulty.features import minilm
from edusync_c2.difficulty.features.minilm import (
    MiniLMConfig,
    MiniLMEncoder,
    SemanticModelError,
    erf_approx,
    gelu,
    is_lfs_pointer,
    read_safetensors,
    required_tensors,
)
from edusync_c2.difficulty.features.wordpiece import WordPieceTokenizer

SRC_DIR = Path(__file__).resolve().parents[2] / "src"
GOLDEN = Path(__file__).resolve().parent / "golden" / "minilm_reference.json"
REAL_DIR = (
    Path(__file__).resolve().parents[3]
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
)

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "the", "##s", "cat", "sat", "on"]
TINY = MiniLMConfig(
    hidden_size=8,
    num_layers=2,
    num_heads=2,
    intermediate_size=16,
    layer_norm_eps=1e-12,
    max_positions=16,
)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Hand-build a safetensors file from (dtype, shape, raw bytes) entries."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    blob = b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [len(blob), len(blob) + len(raw)],
        }
        blob += raw
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + blob)


def f32(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    data = np.ascontiguousarray(array, dtype="<f4")
    return "F32", tuple(data.shape), data.tobytes()


def fake_weights(config: MiniLMConfig = TINY, vocab_rows: int = len(VOCAB), seed: int = 7):
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name, shape in required_tensors(config).items():
        free = {"word": vocab_rows, "position": config.max_positions, "token_type": 2}
        dims = tuple(
            next(rows for key, rows in free.items() if key in name) if d is None else d
            for d in shape
        )
        base = np.ones(dims) if name.endswith("LayerNorm.weight") else np.zeros(dims)
        out[name] = (base + rng.normal(0.0, 0.4, dims)).astype(np.float32)
    return out


@pytest.fixture
def tokenizer() -> WordPieceTokenizer:
    return WordPieceTokenizer(VOCAB)


@pytest.fixture
def encoder(tokenizer) -> MiniLMEncoder:
    return MiniLMEncoder(fake_weights(), TINY, tokenizer, max_length=12)


# ---------------------------------------------------------------- safetensors reader


def test_read_safetensors_roundtrip_and_filter(tmp_path):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.array([1.5, -2.5], dtype=np.float32)
    path = tmp_path / "w.safetensors"
    write_safetensors(
        path,
        {"a": f32(a), "ids": ("I64", (2,), b"\x00" * 16), "b": f32(b), "empty": f32(np.zeros(0))},
    )
    got = read_safetensors(path, {"a", "b", "empty"})
    assert sorted(got) == ["a", "b", "empty"]
    np.testing.assert_array_equal(got["a"], a)
    np.testing.assert_array_equal(got["b"], b)
    assert got["a"].dtype == np.float32
    assert got["empty"].shape == (0,)
    assert sorted(read_safetensors(path, {"b"})) == ["b"]
    assert sorted(read_safetensors(path, set())) == []


def test_unsupported_dtype_only_fails_when_requested(tmp_path):
    path = tmp_path / "w.safetensors"
    write_safetensors(path, {"ok": f32(np.ones(2)), "half": ("F16", (2,), b"\x00" * 4)})
    assert sorted(read_safetensors(path, {"ok"})) == ["ok"]
    with pytest.raises(SemanticModelError, match="unsupported dtype") as info:
        read_safetensors(path)
    assert info.value.reason == "corrupt"


def raw_file(path: Path, header: Any, blob: bytes = b"", declared: int | None = None) -> Path:
    encoded = header if isinstance(header, bytes) else json.dumps(header).encode()
    length = len(encoded) if declared is None else declared
    path.write_bytes(struct.pack("<Q", length) + encoded + blob)
    return path


def test_corrupt_files_are_rejected(tmp_path):
    good = {"t": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    cases = {
        "short": tmp_path / "short",
        "oversize_header": raw_file(tmp_path / "oversize", good, b"\0" * 8, declared=10**9),
        "not_json": raw_file(tmp_path / "nj", b"{nope"),
        "not_utf8": raw_file(tmp_path / "u8", b"\xff\xfe"),
        "not_object": raw_file(tmp_path / "no", [1, 2]),
        "no_dtype": raw_file(tmp_path / "nd", {"t": {"shape": [2], "data_offsets": [0, 8]}}),
        "bad_entry_type": raw_file(tmp_path / "be", {"t": 5}),
        "bad_offsets": raw_file(tmp_path / "bo", {"t": {**good["t"], "data_offsets": [8, 0]}}),
        "negative_shape": raw_file(tmp_path / "ns", {"t": {**good["t"], "shape": [-2]}}),
        "size_mismatch": raw_file(tmp_path / "sm", {"t": {**good["t"], "shape": [3]}}, b"\0" * 8),
        "truncated": raw_file(tmp_path / "tr", good, b"\0" * 4),
    }
    cases["short"].write_bytes(b"\x01\x02")
    for label, path in cases.items():
        with pytest.raises(SemanticModelError) as info:
            read_safetensors(path)
        assert info.value.reason == "corrupt", label


# ---------------------------------------------------------------- pointer detection


def test_lfs_pointer_detection(tmp_path):
    pointer = tmp_path / "pointer"
    pointer.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 9\n")
    padded = tmp_path / "padded"
    padded.write_bytes(b"version https://git-lfs.github.com/spec/v1\n" + b"x" * 2000)
    tiny = tmp_path / "tiny"
    tiny.write_bytes(b"\x00" * 100)
    real = tmp_path / "real"
    real.write_bytes(b"\x00" * 2000)
    assert is_lfs_pointer(pointer)
    assert is_lfs_pointer(padded)
    assert is_lfs_pointer(tiny)
    assert not is_lfs_pointer(real)


# ---------------------------------------------------------------- erf and gelu


def test_erf_approximation_accuracy():
    grid64 = np.linspace(-8.0, 8.0, 4001)
    exact = np.array([math.erf(float(v)) for v in grid64])
    assert np.max(np.abs(erf_approx(grid64) - exact)) < 1.5e-7  # the method's own error bound
    grid = grid64.astype(np.float32)
    got = erf_approx(grid)
    assert got.dtype == np.float32
    assert np.max(np.abs(got - erf_approx(grid64))) < 5e-7  # float32 rounding on top
    assert erf_approx(np.zeros(1, dtype=np.float32))[0] == pytest.approx(0.0, abs=1e-7)


def test_gelu_matches_exact_definition():
    grid = np.linspace(-6.0, 6.0, 2001, dtype=np.float32)
    exact = np.array([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in grid.astype(float)])
    assert np.all(np.abs(gelu(grid) - exact) < 2e-7 * (1.0 + np.abs(grid)))


# ---------------------------------------------------------------- wordpiece ids


def test_tokenizer_ids(tokenizer):
    assert tokenizer.ids(["[UNK]", "[CLS]", "[SEP]", "##s"]) == [1, 2, 3, 7]
    assert tokenizer.ids([]) == []
    with pytest.raises(KeyError):
        tokenizer.ids(["nope"])


def test_vendored_vocab_special_ids():
    real = WordPieceTokenizer.from_resources()
    assert real.ids(["[UNK]", "[CLS]", "[SEP]"]) == [100, 101, 102]


# ---------------------------------------------------------------- config


def write_config(path: Path, **overrides: Any) -> Path:
    body: dict[str, Any] = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 16,
        "hidden_act": "gelu",
    }
    body.update(overrides)
    path.write_text(json.dumps(body))
    return path


def test_config_roundtrip(tmp_path):
    assert MiniLMConfig.from_file(write_config(tmp_path / "c.json")) == TINY


def test_config_errors(tmp_path):
    with pytest.raises(SemanticModelError) as missing:
        MiniLMConfig.from_file(tmp_path / "absent.json")
    assert missing.value.reason == "missing"
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{")
    with pytest.raises(SemanticModelError) as bad_json:
        MiniLMConfig.from_file(garbage)
    assert bad_json.value.reason == "corrupt"
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"hidden_size": 8}))
    with pytest.raises(SemanticModelError) as bad_fields:
        MiniLMConfig.from_file(partial)
    assert bad_fields.value.reason == "corrupt"
    for overrides in ({"hidden_act": "relu"}, {"num_attention_heads": 3}, {"hidden_size": 0}):
        with pytest.raises(SemanticModelError) as info:
            MiniLMConfig.from_file(write_config(tmp_path / "x.json", **overrides))
        assert info.value.reason == "incompatible"


# ---------------------------------------------------------------- synthetic model


def reference_encode(weights, config, tokenizer, text, max_length):
    """Slow float64 loop implementation, written independently of the vectorised encoder."""
    ids = [2, *tokenizer.ids(tokenizer.tokenize(text)[: max_length - 2]), 3]
    w = {k: v.astype(np.float64) for k, v in weights.items()}
    erf = np.vectorize(math.erf)

    def ln(x, wt, bt):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / np.sqrt(var + config.layer_norm_eps) * wt + bt

    x = np.stack(
        [
            w["embeddings.word_embeddings.weight"][t]
            + w["embeddings.position_embeddings.weight"][i]
            + w["embeddings.token_type_embeddings.weight"][0]
            for i, t in enumerate(ids)
        ]
    )
    x = ln(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])
    heads, hid = config.num_heads, config.hidden_size
    dh = hid // heads
    for n in range(config.num_layers):
        p = f"encoder.layer.{n}."

        def lin(name, v, p=p):
            return v @ w[p + name + ".weight"].T + w[p + name + ".bias"]

        q, k, v = (lin(f"attention.self.{s}", x) for s in ("query", "key", "value"))
        ctx = np.zeros_like(x)
        for h in range(heads):
            sl = slice(h * dh, (h + 1) * dh)
            sc = q[:, sl] @ k[:, sl].T / math.sqrt(dh)
            e = np.exp(sc - sc.max(-1, keepdims=True))
            ctx[:, sl] = (e / e.sum(-1, keepdims=True)) @ v[:, sl]
        x = ln(
            x + lin("attention.output.dense", ctx),
            w[p + "attention.output.LayerNorm.weight"],
            w[p + "attention.output.LayerNorm.bias"],
        )
        mid = lin("intermediate.dense", x)
        mid = 0.5 * mid * (1.0 + erf(mid / math.sqrt(2.0)))
        x = ln(
            x + lin("output.dense", mid),
            w[p + "output.LayerNorm.weight"],
            w[p + "output.LayerNorm.bias"],
        )
    pooled = x.mean(0)
    return pooled / np.linalg.norm(pooled)


def test_matches_independent_reference(encoder, tokenizer):
    texts = ["hello world", "The cats sat on the world", "zzz hello", ""]
    got = encoder.encode(texts)
    for row, text in zip(got, texts, strict=True):
        want = reference_encode(fake_weights(), TINY, tokenizer, text, 12)
        np.testing.assert_allclose(row, want, atol=2e-5)


def test_shape_dtype_and_unit_norm(encoder):
    out = encoder.encode(["hello world", "the cat sat on the world", ""])
    assert out.shape == (3, 8)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert encoder.dims == 8
    empty = encoder.encode([])
    assert empty.shape == (0, 8)


def test_deterministic_and_batch_independent(encoder, tokenizer):
    alone = encoder.encode(["hello world"])[0]
    in_batch = encoder.encode(["the cat sat on the world cat", "hello world", "cat"])[1]
    again = MiniLMEncoder(fake_weights(), TINY, tokenizer, max_length=12).encode(["hello world"])[0]
    assert np.array_equal(alone, in_batch)
    assert np.array_equal(alone, again)


def test_order_and_content_matter(encoder):
    a, b, c = encoder.encode(["cat sat", "sat cat", "world world"])
    assert not np.allclose(a, b, atol=1e-4)
    assert not np.allclose(a, c, atol=1e-4)


def test_unknown_words_share_the_unk_id(encoder):
    unk_a, unk_b, known = encoder.encode(["zzz", "qqq", "world"])
    assert np.array_equal(unk_a, unk_b)
    assert not np.allclose(unk_a, known, atol=1e-4)


def test_truncation_keeps_specials_and_limits_length(tokenizer):
    short = MiniLMEncoder(fake_weights(), TINY, tokenizer, max_length=5)
    long_text = "cat sat on the world hello cat sat"
    first_three = "cat sat on"
    assert np.array_equal(short.encode([long_text]), short.encode([first_three]))
    assert not np.array_equal(short.encode([long_text]), short.encode(["cat sat"]))


def test_empty_text_is_cls_sep_only(encoder):
    assert np.array_equal(encoder.encode([""]), encoder.encode(["   "]))
    assert np.isfinite(encoder.encode([""])).all()


def test_encoder_rejects_incompatible_weights(tokenizer):
    full = fake_weights()
    missing = {k: v for k, v in full.items() if k != "encoder.layer.1.output.dense.bias"}
    wrong = {**full, "encoder.layer.0.intermediate.dense.weight": np.zeros((3, 3), np.float32)}
    wrong_rank = {**full, "embeddings.LayerNorm.bias": np.zeros((8, 1), np.float32)}
    for weights in (missing, wrong, wrong_rank):
        with pytest.raises(SemanticModelError) as info:
            MiniLMEncoder(weights, TINY, tokenizer, max_length=12)
        assert info.value.reason == "incompatible"
    few_rows = fake_weights(vocab_rows=len(VOCAB) - 1)
    with pytest.raises(SemanticModelError, match="fewer rows"):
        MiniLMEncoder(few_rows, TINY, tokenizer, max_length=12)
    for bad_length in (2, 17):
        with pytest.raises(SemanticModelError, match="max_length"):
            MiniLMEncoder(full, TINY, tokenizer, max_length=bad_length)
    with pytest.raises(SemanticModelError, match=r"\[CLS\]"):
        MiniLMEncoder(full, TINY, WordPieceTokenizer(VOCAB[:2] + VOCAB[4:]), max_length=12)


# ---------------------------------------------------------------- from_directory


@pytest.fixture
def model_dir(tmp_path) -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    write_config(directory / "config.json")
    vocab_rows = WordPieceTokenizer.from_resources().vocab_size
    weights = fake_weights(vocab_rows=vocab_rows)
    tensors = {name: f32(value) for name, value in weights.items()}
    tensors["embeddings.position_ids"] = ("I64", (1, 4), b"\x00" * 32)
    write_safetensors(directory / "model.safetensors", tensors)
    return directory


def test_from_directory_loads_and_encodes(model_dir):
    encoder = MiniLMEncoder.from_directory(model_dir, max_length=16)
    out = encoder.encode(["Hello, world!", "unknownwordxyz"])
    assert out.shape == (2, 8)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


def test_from_directory_failure_reasons(model_dir, tmp_path):
    with pytest.raises(SemanticModelError) as absent:
        MiniLMEncoder.from_directory(tmp_path / "nowhere")
    assert absent.value.reason == "missing"
    weights = model_dir / "model.safetensors"
    good = weights.read_bytes()
    weights.write_bytes(good[: len(good) // 2])
    with pytest.raises(SemanticModelError) as truncated:
        MiniLMEncoder.from_directory(model_dir)
    assert truncated.value.reason == "corrupt"
    weights.write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
    with pytest.raises(SemanticModelError) as pointer:
        MiniLMEncoder.from_directory(model_dir)
    assert pointer.value.reason == "lfs_pointer"
    weights.write_bytes(good)
    (model_dir / "config.json").unlink()
    with pytest.raises(SemanticModelError) as no_config:
        MiniLMEncoder.from_directory(model_dir)
    assert no_config.value.reason == "missing"


def test_vendored_vocab_is_identical_to_model_vocab():
    model_vocab = REAL_DIR / "vocab.txt"
    if not model_vocab.is_file():
        pytest.skip("model directory not present")
    vendored = SRC_DIR / "edusync_c2" / "difficulty" / "resources" / "bert_uncased_vocab.txt"
    assert model_vocab.read_bytes() == vendored.read_bytes()


# ---------------------------------------------------------------- importing without numpy


def test_package_imports_without_numpy():
    code = (
        "import sys; sys.modules['numpy'] = None\n"
        "import edusync_c2\n"
        "import edusync_c2.difficulty.features.minilm as m\n"
        "print(m.DEFAULT_MAX_LENGTH)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_DIR), *sys.path])
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == str(minilm.DEFAULT_MAX_LENGTH)


# ---------------------------------------------------------------- real weights (ml marker)


def real_weights_available() -> bool:
    weights = REAL_DIR / "model.safetensors"
    return weights.is_file() and not is_lfs_pointer(weights)


@pytest.mark.ml
def test_real_model_matches_reference_embeddings():
    if not real_weights_available():
        pytest.skip("real MiniLM weights are missing or an LFS pointer")
    if not GOLDEN.is_file():
        pytest.skip("minilm_reference.json has not been generated yet")
    cases = json.loads(GOLDEN.read_text(encoding="utf-8"))["cases"]
    encoder = MiniLMEncoder.from_directory(REAL_DIR)
    assert encoder.dims == 384
    for case in cases:
        got = encoder.encode([case["text"]])[0]
        want = np.asarray(case["embedding"], dtype=np.float64)
        cosine = float(np.dot(got, want) / (np.linalg.norm(got) * np.linalg.norm(want)))
        assert cosine >= 0.9999, case["text"]


@pytest.mark.ml
def test_real_model_speed_and_sanity():
    if not real_weights_available():
        pytest.skip("real MiniLM weights are missing or an LFS pointer")
    encoder = MiniLMEncoder.from_directory(REAL_DIR)
    text = "A hash table maps keys to buckets so lookups take constant time on average."
    encoder.encode([text])
    start = time.perf_counter()
    vec = encoder.encode([text])[0]
    assert time.perf_counter() - start < 1.0
    near, far = encoder.encode(["Hash maps give fast average lookups.", "I ate a sandwich."])
    assert float(vec @ near) > float(vec @ far)
