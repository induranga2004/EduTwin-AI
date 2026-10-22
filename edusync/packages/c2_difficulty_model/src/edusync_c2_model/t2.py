"""T2: frozen MiniLM embedding plus five simple text features into a closed-form ridge regressor.

``T2Model`` predicts the z-scored difficulty (higher = harder, scaled on the training part only) of
a text. Input = the sentence embedding (unit norm, 384 dims) concatenated with the five features of
``baselines_text`` (Flesch-Kincaid, word and sentence length, rare-word share, fragmentation); all
columns are standardised on TRAIN only inside ``RidgeModel``. Alpha comes from a seeded grouped
k-fold on train. numpy only; the encoder is any ``texts -> (n, d)`` callable.

Embedding variants (``chunk_words``):

* ``None``  - the encoder sees each text once and truncates it to its token limit (128 tokens, about
  the first 90 words). Kept for comparison.
* ``int``   - the text is cut into windows of at most that many words (on sentence boundaries where
  possible), every window is embedded, the window vectors are mean-pooled and L2-normalised.

Uncertainty proxy (not a probability): ``cv_rmse * max(1, d / d_median)`` where ``cv_rmse`` is the
out-of-fold RMSE of the chosen alpha on train and ``d`` is the distance of the text embedding from
the train embedding centroid (``d_median`` = the median train distance). Texts that look unlike the
training texts get a larger value.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from edusync_c2_model.baselines_text import (
    FEATURE_NAMES,
    RidgeModel,
    feature_matrix,
    fold_assignment,
    split_sentences,
)
from edusync_c2_model.models.embeddings import Embedder

FORMAT_VERSION = 1
MODEL_KIND = "t2_ridge"
DEFAULT_CHUNK_WORDS = 100
ALPHA_GRID = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
CV_FOLDS = 5
LICENCE_NOTE = (
    "Trained on CLEAR (CC BY-NC-SA 4.0) + OneStopEnglish (CC BY-SA 4.0): research/non-commercial."
)
NPZ_NAME = "t2.npz"
CONFIG_NAME = "t2_config.json"


class T2Error(ValueError):
    """The model cannot be fitted, loaded or applied."""


def chunk_text(text: str, max_words: int) -> list[str]:
    """Windows of at most ``max_words`` words, breaking at sentence ends when possible."""
    if max_words < 1:
        raise T2Error("max_words must be at least 1")
    chunks: list[str] = []
    buffer: list[str] = []
    for sentence in split_sentences(text):
        words = sentence.split()
        if buffer and len(buffer) + len(words) > max_words:
            chunks.append(" ".join(buffer))
            buffer = []
        for start in range(0, len(words), max_words):
            piece = words[start : start + max_words]
            if len(buffer) + len(piece) > max_words:
                chunks.append(" ".join(buffer))
                buffer = []
            buffer.extend(piece)
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks or [text]


def _unit(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norm < 1e-12, 1.0, norm)


def embed_texts(embed: Embedder, texts: Sequence[str], chunk_words: int | None) -> np.ndarray:
    """One unit-norm vector per text (see the module docstring for the two variants)."""
    if not texts:
        return np.zeros((0, 0))
    if chunk_words is None:
        return _unit(np.asarray(embed(list(texts)), dtype=np.float64))
    pieces = [chunk_text(t, chunk_words) for t in texts]
    flat = np.asarray(embed([c for chunks in pieces for c in chunks]), dtype=np.float64)
    pooled = np.empty((len(texts), flat.shape[1]))
    start = 0
    for row, chunks in enumerate(pieces):
        pooled[row] = flat[start : start + len(chunks)].mean(axis=0)
        start += len(chunks)
    return _unit(pooled)


def _select_alpha(
    x: np.ndarray, y: np.ndarray, groups: Sequence[str], alphas: Sequence[float], seed: int
) -> tuple[float, float]:
    """``(alpha, out-of-fold RMSE)``: lowest grouped k-fold error, ties go to the smaller alpha."""
    assigned = fold_assignment(groups, CV_FOLDS, seed)
    best_alpha, best_sse, best_count = alphas[0], np.inf, 0
    for alpha in alphas:
        sse, count = 0.0, 0
        for fold in range(CV_FOLDS):
            held = assigned == fold
            if held.all() or not held.any():
                continue
            model = RidgeModel(alpha).fit(x[~held], y[~held])
            sse += float(((model.predict(x[held]) - y[held]) ** 2).sum())
            count += int(held.sum())
        if sse < best_sse:
            best_alpha, best_sse, best_count = alpha, sse, count
    if best_count == 0:  # a single group: no honest out-of-fold estimate exists
        fitted = RidgeModel(best_alpha).fit(x, y).predict(x)
        return float(best_alpha), float(np.sqrt(np.mean((fitted - y) ** 2)))
    return float(best_alpha), float(np.sqrt(best_sse / best_count))


class T2Model:
    """Embedding + simple features -> ridge. Build with ``fit`` or ``from_npz_dict``."""

    def __init__(self, embed: Embedder | None, encoder_sha: str, chunk_words: int | None) -> None:
        self.embed = embed
        self.encoder_sha = encoder_sha
        self.chunk_words = chunk_words
        self.ridge = RidgeModel(1.0)
        self.emb_mean = np.zeros(0)
        self.median_dist = 1.0
        self.cv_rmse = 0.0
        self.alpha = 1.0
        self.n_train = 0
        self.seed = 0

    # ------------------------------------------------------------------ features

    def _embedding(self, texts: Sequence[str]) -> np.ndarray:
        if self.embed is None:
            raise T2Error("this model was loaded without an embedder")
        return embed_texts(self.embed, texts, self.chunk_words)

    @staticmethod
    def _design(embedding: np.ndarray, texts: Sequence[str]) -> np.ndarray:
        return np.hstack([embedding, feature_matrix(list(texts))])

    # ------------------------------------------------------------------ training

    def fit(
        self,
        texts: Sequence[str],
        targets: Sequence[float],
        seed: int,
        groups: Sequence[str] | None = None,
        alphas: Sequence[float] = ALPHA_GRID,
    ) -> "T2Model":
        """Fit on z-scored difficulty ``targets``; ``groups`` keeps the CV folds clean."""
        y = np.asarray(targets, dtype=np.float64)
        if len(texts) != y.size or (groups is not None and len(groups) != y.size):
            raise T2Error("texts, targets and groups must have the same length")
        if y.size < 2 or not np.all(np.isfinite(y)):
            raise T2Error("need at least 2 texts and finite targets")
        labels = [str(i) for i in range(y.size)] if groups is None else list(groups)
        embedding = self._embedding(texts)
        x = self._design(embedding, texts)
        self.alpha, self.cv_rmse = _select_alpha(x, y, labels, alphas, seed)
        self.ridge = RidgeModel(self.alpha).fit(x, y)
        self.emb_mean = embedding.mean(axis=0)
        distances = np.linalg.norm(embedding - self.emb_mean, axis=1)
        median = float(np.median(distances))
        self.median_dist = median if median > 1e-12 else 1.0
        self.n_train = int(y.size)
        self.seed = int(seed)
        return self

    # ------------------------------------------------------------------ inference

    def predict(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """``(difficulty z-score, uncertainty proxy)`` per text, both float arrays."""
        if not texts:
            return np.zeros(0), np.zeros(0)
        embedding = self._embedding(texts)
        z = self.ridge.predict(self._design(embedding, texts))
        distance = np.linalg.norm(embedding - self.emb_mean, axis=1)
        return z, self.cv_rmse * np.maximum(1.0, distance / self.median_dist)

    # ------------------------------------------------------------------ persistence

    def config(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "model_kind": MODEL_KIND,
            "encoder_sha": self.encoder_sha,
            "encoder_max_tokens": 128,
            "chunk_words": self.chunk_words,
            "embedding_dims": int(self.emb_mean.size),
            "simple_feature_names": list(FEATURE_NAMES),
            "alpha": self.alpha,
            "alpha_grid": list(ALPHA_GRID),
            "cv_folds": CV_FOLDS,
            "cv_rmse": self.cv_rmse,
            "median_dist": self.median_dist,
            "n_train": self.n_train,
            "seed": self.seed,
            "target": "z-scored difficulty, higher = harder (train-only mean/std)",
            "uncertainty": "cv_rmse * max(1, distance_to_train_centroid / median_train_distance)",
            "licence_note": LICENCE_NOTE,
        }

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Plain arrays for ``np.savez`` (no pickles); the config rides along as a JSON string."""
        return {
            "config_json": np.array(json.dumps(self.config(), sort_keys=True)),
            "x_mean": self.ridge.mean,
            "x_std": self.ridge.std,
            "coef": self.ridge.coef,
            "intercept": np.array([self.ridge.intercept]),
            "emb_mean": self.emb_mean,
        }

    @classmethod
    def from_npz_dict(cls, data, embed: Embedder | None = None) -> "T2Model":
        """Rebuild from ``to_npz_dict`` output or a loaded ``.npz``; ``embed`` is for predict."""
        try:
            config = json.loads(str(data["config_json"]))
            arrays = {k: np.asarray(data[k], dtype=np.float64) for k in _ARRAY_KEYS}
        except (KeyError, ValueError) as exc:
            raise T2Error(f"invalid T2 artifact: {exc}") from exc
        if config.get("model_kind") != MODEL_KIND or config.get("format_version") != FORMAT_VERSION:
            raise T2Error("unsupported T2 artifact version or kind")
        dims = len(FEATURE_NAMES) + int(config["embedding_dims"])
        if (
            arrays["x_mean"].shape != (dims,)
            or arrays["x_std"].shape != (dims,)
            or arrays["coef"].shape != (dims,)
            or arrays["intercept"].shape != (1,)
            or arrays["emb_mean"].shape != (dims - len(FEATURE_NAMES),)
        ):
            raise T2Error("T2 artifact arrays do not match its config")
        model = cls(embed, config["encoder_sha"], config["chunk_words"])
        model.alpha = float(config["alpha"])
        model.cv_rmse = float(config["cv_rmse"])
        model.median_dist = float(config["median_dist"])
        model.n_train = int(config["n_train"])
        model.seed = int(config["seed"])
        model.ridge = RidgeModel(model.alpha)
        model.ridge.mean, model.ridge.std = arrays["x_mean"], arrays["x_std"]
        model.ridge.coef = arrays["coef"]
        model.ridge.intercept = float(arrays["intercept"][0])
        model.emb_mean = arrays["emb_mean"]
        return model

    def save(self, directory: Path) -> tuple[Path, Path]:
        """Write ``t2.npz`` and ``t2_config.json`` into ``directory`` and return both paths."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        npz, config = directory / NPZ_NAME, directory / CONFIG_NAME
        np.savez(npz, **self.to_npz_dict())
        config.write_text(
            json.dumps(self.config(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return npz, config

    @classmethod
    def load(cls, directory: Path, embed: Embedder | None = None) -> "T2Model":
        try:
            with np.load(Path(directory) / NPZ_NAME, allow_pickle=False) as stored:
                return cls.from_npz_dict(dict(stored), embed)
        except OSError as exc:
            raise T2Error(f"cannot read the T2 artifact: {exc}") from exc


_ARRAY_KEYS = ("x_mean", "x_std", "coef", "intercept", "emb_mean")
