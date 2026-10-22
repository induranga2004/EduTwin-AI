"""Process-wide lazy loader of the MiniLM encoder (success and failure are both cached).

The first request for a model directory loads the weights once, even under a thread race; later
requests reuse the result. A failure is remembered too, so a broken or missing model costs one
attempt and one WARNING, never one per request. Logs carry the reason class and a short weights
fingerprint, never paths or text.
"""

import hashlib
import threading
from collections.abc import Callable
from pathlib import Path

from edusync_c2.difficulty.features.minilm import (
    WEIGHTS_FILE,
    MiniLMEncoder,
    SemanticModelError,
)
from edusync_c2.logger import get_logger

__all__ = [
    "EncoderLoader",
    "SemanticModelError",
    "get_default_encoder",
    "reset_default_encoder",
    "weights_fingerprint",
]

logger = get_logger(__name__)

EncoderLoader = Callable[[Path], MiniLMEncoder]
_FINGERPRINT_BYTES = 1 << 20


def weights_fingerprint(weights: Path) -> str:
    """First 8 hex chars of sha256(first 1 MiB of the file + its size), cheap for 90 MB files."""
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        digest.update(handle.read(_FINGERPRINT_BYTES))
    digest.update(str(weights.stat().st_size).encode())
    return digest.hexdigest()[:8]


class _Cache:
    """Per resolved model directory: the encoder, or None when loading failed."""

    def __init__(self) -> None:
        self.entries: dict[str, MiniLMEncoder | None] = {}


_cache = _Cache()
_lock = threading.Lock()


def _load(model_dir: Path, loader: EncoderLoader) -> MiniLMEncoder | None:
    try:
        encoder = loader(model_dir)
        encoder.sha = weights_fingerprint(model_dir / WEIGHTS_FILE)
    except Exception as exc:
        reason = exc.reason if isinstance(exc, SemanticModelError) else type(exc).__name__
        logger.warning("semantic_unavailable reason=%s", reason)
        return None
    logger.info("semantic_loaded sha=%s dims=%d", encoder.sha, encoder.dims)
    return encoder


def get_default_encoder(
    model_dir: Path | None, *, loader: EncoderLoader | None = None
) -> MiniLMEncoder | None:
    """The shared encoder for `model_dir`; None (cached) if it cannot be loaded.

    `model_dir=None` means the scorer is not configured: None is returned without logging.
    `loader` replaces `MiniLMEncoder.from_directory` (for tests) on the first load of a directory.
    """
    if model_dir is None:
        return None
    key = str(model_dir.resolve())
    if key not in _cache.entries:
        with _lock:
            if key not in _cache.entries:
                _cache.entries[key] = _load(model_dir, loader or MiniLMEncoder.from_directory)
    return _cache.entries[key]


def reset_default_encoder() -> None:
    """Forget every cached encoder or failure so the next call loads again (for tests)."""
    with _lock:
        _cache.entries.clear()
