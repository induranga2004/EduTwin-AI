"""Text-difficulty models T1 (fine-tuned MiniLM, ONNX) and T2 (frozen MiniLM + ridge, numpy).

Both predict a z-scored difficulty (higher = harder) of a text and map it to a percentile against
the distribution of the same model's predictions over the public training texts (stored in the
artifact). Trained on general English, not lecture slides. onnxruntime and tokenizers are imported
lazily and only by the default factories; tests inject fakes.
"""

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from edusync_c2.difficulty.features.vocab import VocabularyAnalyzer, get_default_analyzer
from edusync_c2.difficulty.scoring.reference import ScoringError, percentile_in

T2_NPZ = "t2.npz"
T2_CONFIG = "t2_config.json"
T2_FORMAT_VERSION = 1
T2_KIND = "t2_ridge"
T2_FEATURES = 5  # fk_grade, avg_word_len, avg_sentence_len, rare_word_share, fragmentation
T1_INT8 = "t1_int8.onnx"
T1_FP32 = "t1_fp32.onnx"
T1_TOKENIZER = "tokenizer.json"
T1_CONFIG = "inference_config.json"
T1_BATCH = 16
TEXT_CAP = 1500  # characters of a concept's slide text given to the models

Embed = Callable[[Sequence[str]], np.ndarray]
SessionFactory = Callable[[Path], Any]  # path -> object with .run(None, feeds) -> [array]
TokenizerFactory = Callable[[Path, int], Callable[[list[str]], list[list[int]]]]

_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SENTENCE_END = re.compile(r"[.!?]+(?:\s+|$)")
_VOWELS = re.compile(r"[aeiouy]+")
_KEEP_E = ("le", "ee", "ye", "ue")
_KEEP_ES_ED = ("ses", "zes", "xes", "ches", "shes", "ges", "ces", "ted", "ded")


# ----------------------------------------------------------------- T2 (ported predict path)


def _syllables(word: str) -> int:
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return 0
    groups = len(_VOWELS.findall(letters))
    silent_e = letters.endswith("e") and not letters.endswith(_KEEP_E)
    silent_es_ed = letters.endswith(("es", "ed")) and not letters.endswith(_KEEP_ES_ED)
    if (silent_e or silent_es_ed) and groups > 1:
        groups -= 1
    return max(1, groups)


def _sentences(text: str) -> list[str]:
    return [part for part in _SENTENCE_END.split(text) if part.strip()]


def simple_features(text: str, analyzer: VocabularyAnalyzer) -> np.ndarray:
    """The five simple T2 features (same definitions as the trainer; zeros without words)."""
    words = _WORD.findall(text)
    if not words:
        return np.zeros(T2_FEATURES)
    stats = analyzer.analyze(text)
    rare, fragmentation = (
        (0.0, 1.0) if stats is None else (stats.unfamiliar_share, stats.fragmentation)
    )
    sentences = max(1, len(_sentences(text)))
    syllables = sum(_syllables(w) for w in words)
    grade = 0.39 * len(words) / sentences + 11.8 * syllables / len(words) - 15.59
    return np.array(
        [
            grade,
            sum(len(w) for w in words) / len(words),
            len(words) / sentences,
            rare,
            fragmentation,
        ]
    )


def chunk_text(text: str, max_words: int) -> list[str]:
    """Windows of at most ``max_words`` words, breaking at sentence ends when possible."""
    chunks: list[str] = []
    buffer: list[str] = []
    for sentence in _sentences(text):
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


def embed_texts(embed: Embed, texts: Sequence[str], chunk_words: int | None) -> np.ndarray:
    """One unit-norm vector per text: windows embedded, mean-pooled, normalised."""
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


class T2Scorer:
    """Numpy ridge over the MiniLM embedding and five simple features."""

    name = "t2"

    def __init__(
        self,
        embed: Embed,
        arrays: dict[str, np.ndarray],
        chunk_words: int | None,
        z_table: Sequence[float],
        analyzer: VocabularyAnalyzer,
    ) -> None:
        self.embed = embed
        self.arrays = arrays
        self.chunk_words = chunk_words
        self.z_table = list(z_table)
        self.analyzer = analyzer

    @classmethod
    def load(cls, directory: Path, embed: Embed, z_table: Sequence[float]) -> "T2Scorer":
        directory = Path(directory)
        try:
            with np.load(directory / T2_NPZ, allow_pickle=False) as stored:
                config = json.loads(str(stored["config_json"]))
                arrays = {
                    k: np.asarray(stored[k], dtype=np.float64)
                    for k in ("x_mean", "x_std", "coef", "intercept", "emb_mean")
                }
            dims = T2_FEATURES + int(config["embedding_dims"])
            valid = (
                config.get("model_kind") == T2_KIND
                and config.get("format_version") == T2_FORMAT_VERSION
                and arrays["x_mean"].shape == (dims,)
                and arrays["x_std"].shape == (dims,)
                and arrays["coef"].shape == (dims,)
                and arrays["intercept"].shape == (1,)
            )
            chunk = config["chunk_words"]
            chunk_words = None if chunk is None else int(chunk)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ScoringError(f"cannot load T2: {type(exc).__name__}") from exc
        if not valid:
            raise ScoringError("unsupported or inconsistent T2 artifact")
        analyzer = get_default_analyzer()
        if analyzer is None:
            raise ScoringError("vocabulary resources are unavailable for T2")
        return cls(embed, arrays, chunk_words, z_table, analyzer)

    def z(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        embedding = embed_texts(self.embed, texts, self.chunk_words)
        simple = np.vstack([simple_features(t, self.analyzer) for t in texts])
        x = np.hstack([embedding, simple])
        a = self.arrays
        return ((x - a["x_mean"]) / a["x_std"]) @ a["coef"] + float(a["intercept"][0])

    def percentiles(self, texts: Sequence[str]) -> list[float]:
        return [percentile_in(self.z_table, float(v)) or 0.0 for v in self.z(texts)]


# ----------------------------------------------------------------------- T1 (ONNX INT8)


def default_session_factory(path: Path) -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def default_tokenizer_factory(
    path: Path, max_length: int
) -> Callable[[list[str]], list[list[int]]]:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=max_length)

    def encode(texts: list[str]) -> list[list[int]]:
        return [e.ids for e in tokenizer.encode_batch(texts)]

    return encode


class T1Scorer:
    """Fine-tuned MiniLM regression head in ONNX (INT8 by default)."""

    name = "t1"

    def __init__(
        self,
        session: Any,
        encode: Callable[[list[str]], list[list[int]]],
        z_table: Sequence[float],
        model_file: str = T1_INT8,
    ) -> None:
        self.session = session
        self.encode = encode
        self.z_table = list(z_table)
        self.model_file = model_file

    @classmethod
    def load(
        cls,
        directory: Path,
        z_table: Sequence[float],
        *,
        session_factory: SessionFactory = default_session_factory,
        tokenizer_factory: TokenizerFactory = default_tokenizer_factory,
    ) -> "T1Scorer":
        directory = Path(directory)
        model = next((f for f in (T1_INT8, T1_FP32) if (directory / f).is_file()), None)
        if model is None:
            raise ScoringError("no T1 ONNX model in the artifact")
        try:
            config = json.loads((directory / T1_CONFIG).read_text(encoding="utf-8"))
            max_length = int(config["max_length"])
            session = session_factory(directory / model)
            encode = tokenizer_factory(directory / T1_TOKENIZER, max_length)
        except Exception as exc:
            raise ScoringError(f"cannot load T1: {type(exc).__name__}") from exc
        return cls(session, encode, z_table, model)

    def z(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros(len(texts), dtype=np.float64)
        ids = self.encode(list(texts))
        for start in range(0, len(texts), T1_BATCH):
            batch = ids[start : start + T1_BATCH]
            width = max(len(row) for row in batch)
            input_ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for r, row in enumerate(batch):
                input_ids[r, : len(row)] = row
                mask[r, : len(row)] = 1
            feeds = {"input_ids": input_ids, "attention_mask": mask}
            out[start : start + len(batch)] = np.asarray(self.session.run(None, feeds)[0]).reshape(
                -1
            )
        return out

    def percentiles(self, texts: Sequence[str]) -> list[float]:
        return [percentile_in(self.z_table, float(v)) or 0.0 for v in self.z(texts)]
