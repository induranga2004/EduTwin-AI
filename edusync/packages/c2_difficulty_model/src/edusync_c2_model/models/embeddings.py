"""Frozen-encoder embeddings with an on-disk cache keyed by sha256(text) and the encoder identity.

``Embedder`` is any callable ``texts -> (n, d) array`` (the real one wraps ``MiniLMEncoder``; tests
pass a fake). ``CachedEmbedder`` computes each distinct text once, in sorted order, so the result
never depends on call order. The cache file is a plain ``.npz`` (no pickles).
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]


def text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbedder:
    """Memoising wrapper; ``encoder_id`` (e.g. the weights sha) names the cache file."""

    def __init__(self, embed: Embedder, encoder_id: str, cache_dir: Path | None = None) -> None:
        self._embed = embed
        self.encoder_id = encoder_id
        self._path = None if cache_dir is None else Path(cache_dir) / f"embeddings-{encoder_id}.npz"
        self._vectors: dict[str, np.ndarray] = {}
        self.computed = 0  # texts sent to the encoder (a cache-miss counter, for tests and logs)
        if self._path is not None and self._path.is_file():
            with np.load(self._path, allow_pickle=False) as stored:
                for key, vector in zip(stored["keys"].tolist(), stored["vectors"], strict=True):
                    self._vectors[key] = vector

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        keys = [text_key(t) for t in texts]
        missing = sorted(
            {k: t for k, t in zip(keys, texts, strict=True) if k not in self._vectors}.items()
        )
        if missing:
            fresh = np.asarray(self._embed([t for _, t in missing]), dtype=np.float64)
            for (key, _), vector in zip(missing, fresh, strict=True):
                self._vectors[key] = vector
            self.computed += len(missing)
            self._save()
        return np.stack([self._vectors[k] for k in keys]) if keys else np.zeros((0, 0))

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        order = sorted(self._vectors)
        np.savez(
            self._path,
            keys=np.array(order),
            vectors=np.stack([self._vectors[k] for k in order]),
        )


def encoder_embedder(model_dir: Path) -> tuple[Embedder, str]:
    """The frozen MiniLM as ``(embed, weights_sha)``; ``RuntimeError`` when it cannot load."""
    from edusync_c2.difficulty.features.semantic_loader import get_default_encoder

    encoder = get_default_encoder(Path(model_dir))
    if encoder is None:
        raise RuntimeError(f"the MiniLM encoder in {model_dir} cannot be loaded")
    return encoder.encode, str(encoder.sha)
