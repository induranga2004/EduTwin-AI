"""Lazy shared MiniLM loader: single load under a race, cached failures, reasons, privacy."""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from edusync_c2.difficulty.features import semantic_loader
from edusync_c2.difficulty.features.minilm import MiniLMEncoder
from edusync_c2.difficulty.features.semantic_loader import (
    SemanticModelError,
    get_default_encoder,
    reset_default_encoder,
    weights_fingerprint,
)

LOGGER = "edusync_c2.difficulty.features.semantic_loader"


@pytest.fixture(autouse=True)
def fresh_cache():
    reset_default_encoder()
    yield
    reset_default_encoder()


class FakeEncoder:
    dims = 384
    sha: str | None = None


def make_dir(root: Path, name: str, payload: bytes = b"w" * 3000) -> Path:
    directory = root / name
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(payload)
    return directory


class CountingLoader:
    def __init__(self, delay: float = 0.0, error: Exception | None = None) -> None:
        self.calls = 0
        self.delay = delay
        self.error = error
        self.lock = threading.Lock()

    def __call__(self, model_dir: Path):
        with self.lock:
            self.calls += 1
        time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return FakeEncoder()


def lines(caplog, level: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelname == level]


def test_none_dir_is_unavailable_without_logs(caplog):
    caplog.set_level("DEBUG", logger="edusync_c2")
    loader = CountingLoader()
    assert get_default_encoder(None, loader=loader) is None
    assert loader.calls == 0
    assert not [r for r in caplog.records if r.name == LOGGER]


def test_success_loads_once_and_logs_fingerprint(tmp_path, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    directory = make_dir(tmp_path, "m")
    loader = CountingLoader()
    first = get_default_encoder(directory, loader=loader)
    second = get_default_encoder(directory, loader=loader)
    assert first is second
    assert loader.calls == 1
    expected = hashlib.sha256((b"w" * 3000) + b"3000").hexdigest()[:8]
    assert first.sha == expected
    assert lines(caplog, "INFO") == [f"semantic_loaded sha={expected} dims=384"]
    assert not lines(caplog, "WARNING")


def test_fingerprint_reads_only_the_first_mebibyte_and_the_size(tmp_path):
    big = tmp_path / "big.bin"
    head = b"a" * (1 << 20)
    big.write_bytes(head + b"b" * 1000)
    assert weights_fingerprint(big) == hashlib.sha256(head + b"1049576").hexdigest()[:8]
    other = tmp_path / "other.bin"
    other.write_bytes(head + b"c" * 1000)
    assert weights_fingerprint(other) == weights_fingerprint(big)


def test_first_load_is_race_free(tmp_path, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    directory = make_dir(tmp_path, "m")
    loader = CountingLoader(delay=0.05)
    barrier = threading.Barrier(16)

    def work(_: int):
        barrier.wait()
        return get_default_encoder(directory, loader=loader)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(work, range(16)))
    assert loader.calls == 1
    assert all(result is results[0] for result in results)
    assert len(lines(caplog, "INFO")) == 1


def test_failure_is_cached_per_directory_and_logged_once(tmp_path, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    bad = make_dir(tmp_path, "bad")
    good = make_dir(tmp_path, "good")
    failing = CountingLoader(error=ValueError("secret detail"))
    assert get_default_encoder(bad, loader=failing) is None
    assert get_default_encoder(bad, loader=failing) is None
    assert failing.calls == 1
    assert lines(caplog, "WARNING") == ["semantic_unavailable reason=ValueError"]
    working = CountingLoader()
    assert get_default_encoder(good, loader=working) is not None
    assert get_default_encoder(bad, loader=working) is None  # still the cached failure
    assert working.calls == 1
    assert all("secret" not in r.getMessage() for r in caplog.records)


def test_loader_semantic_error_reason_is_logged(tmp_path, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    directory = make_dir(tmp_path, "m")
    error = SemanticModelError("incompatible", "private detail")
    assert get_default_encoder(directory, loader=CountingLoader(error=error)) is None
    assert lines(caplog, "WARNING") == ["semantic_unavailable reason=incompatible"]


def test_fingerprint_failure_is_unavailable(tmp_path, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    empty_dir = tmp_path / "no_weights"
    empty_dir.mkdir()
    assert get_default_encoder(empty_dir, loader=CountingLoader()) is None
    assert lines(caplog, "WARNING") == ["semantic_unavailable reason=FileNotFoundError"]
    assert not lines(caplog, "INFO")


def test_real_loader_reasons(tmp_path, caplog):
    caplog.set_level("INFO", logger="edusync_c2")
    missing = tmp_path / "nowhere"
    pointer = make_dir(
        tmp_path, "pointer", b"version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n"
    )
    corrupt = make_dir(tmp_path, "corrupt", b"\xff" * 5000)
    (corrupt / "config.json").write_text(
        '{"hidden_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,'
        ' "intermediate_size": 16, "layer_norm_eps": 1e-12, "max_position_embeddings": 16,'
        ' "hidden_act": "gelu"}'
    )
    for directory in (missing, pointer, corrupt):
        assert get_default_encoder(directory) is None
    assert lines(caplog, "WARNING") == [
        "semantic_unavailable reason=missing",
        "semantic_unavailable reason=lfs_pointer",
        "semantic_unavailable reason=corrupt",
    ]
    assert not lines(caplog, "INFO")


def test_logs_never_contain_paths(tmp_path, caplog):
    caplog.set_level("DEBUG", logger="edusync_c2")
    secret_dir = make_dir(tmp_path, "very_secret_dir")
    get_default_encoder(secret_dir, loader=CountingLoader())
    get_default_encoder(tmp_path / "very_secret_missing")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "very_secret" not in text
    assert str(tmp_path) not in text


def test_reset_forgets_success_and_failure(tmp_path):
    directory = make_dir(tmp_path, "m")
    loader = CountingLoader()
    get_default_encoder(directory, loader=loader)
    reset_default_encoder()
    get_default_encoder(directory, loader=loader)
    assert loader.calls == 2


def test_same_directory_via_different_spelling_shares_one_entry(tmp_path):
    directory = make_dir(tmp_path, "m")
    loader = CountingLoader()
    get_default_encoder(directory, loader=loader)
    get_default_encoder(tmp_path / "m" / ".." / "m", loader=loader)
    assert loader.calls == 1


def test_default_loader_is_from_directory(monkeypatch, tmp_path):
    directory = make_dir(tmp_path, "m")
    seen: list[Path] = []

    def fake(model_dir: Path):
        seen.append(model_dir)
        return FakeEncoder()

    monkeypatch.setattr(MiniLMEncoder, "from_directory", staticmethod(fake))
    assert get_default_encoder(directory) is not None
    assert seen == [directory]
    assert semantic_loader.SemanticModelError is SemanticModelError
