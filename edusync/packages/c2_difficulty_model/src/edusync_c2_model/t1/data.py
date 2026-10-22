"""Dataset preparation for T1: train-only z-scored targets, tokenization hook, dynamic padding.

The tokenizer is injected as ``encode(texts) -> list[list[int]]`` (already truncated and with the
special tokens), so this module has no dependency on ``tokenizers`` or torch.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.public_data.splits import PARTS
from edusync_c2_model.public_data.targets import fit_difficulty_scaler, to_difficulty

Encode = Callable[[list[str]], list[list[int]]]
PAD_ID = 0


@dataclass(frozen=True)
class Row:
    source: str
    group_id: str
    text: str
    ids: tuple[int, ...]
    target: float  # z-scored difficulty, higher = harder


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray  # int64 [batch, seq]
    attention_mask: np.ndarray  # int64 [batch, seq]
    targets: np.ndarray  # float32 [batch]
    index: np.ndarray  # positions of the rows in the source list


def build_rows(
    parts_by_source: dict[str, dict[str, list[TextExample]]],
    higher_is_easier: dict[str, bool],
    encode: Encode,
    *,
    limit_rows: int | None = None,
) -> tuple[dict[str, list[Row]], dict[str, dict]]:
    """Rows per part for all sources, each source z-scored with its OWN train statistics.

    ``limit_rows`` keeps the first N examples of every part of every source (smoke runs).
    Returns ``(rows_by_part, scalers_by_source)``.
    """
    rows: dict[str, list[Row]] = {part: [] for part in PARTS}
    scalers: dict[str, dict] = {}
    for source in sorted(parts_by_source):
        parts = parts_by_source[source]
        cut = {p: parts[p][:limit_rows] if limit_rows else parts[p] for p in PARTS}
        scaler = fit_difficulty_scaler(
            [e.target for e in cut["train"]], higher_is_easier=higher_is_easier[source]
        )
        scalers[source] = scaler
        for part in PARTS:
            examples = cut[part]
            z = to_difficulty([e.target for e in examples], scaler)
            ids = encode([e.text for e in examples])
            rows[part].extend(
                Row(source, e.group_id, e.text, tuple(i), float(t))
                for e, i, t in zip(examples, ids, z, strict=True)
            )
    return rows, scalers


def pad_batch(
    sequences: Sequence[Sequence[int]],
    targets: Sequence[float] | None = None,
    index: Sequence[int] | None = None,
) -> Batch:
    """Right-pad to the longest sequence in the batch (dynamic padding)."""
    longest = max((len(s) for s in sequences), default=1)
    ids = np.full((len(sequences), longest), PAD_ID, dtype=np.int64)
    mask = np.zeros((len(sequences), longest), dtype=np.int64)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = seq
        mask[row, : len(seq)] = 1
    y = np.zeros(len(sequences), dtype=np.float32) if targets is None else np.asarray(targets)
    pos = np.arange(len(sequences)) if index is None else np.asarray(index)
    return Batch(ids, mask, y.astype(np.float32), pos)


def bucket_batches(
    lengths: Sequence[int], batch_size: int, *, seed: int, shuffle: bool
) -> Iterator[np.ndarray]:
    """Index batches of similar length. With ``shuffle``: sort inside random mega-chunks of
    ``50 * batch_size`` rows, then shuffle the batch order; without: one global length sort."""
    n = len(lengths)
    arr = np.asarray(lengths)
    if not shuffle:
        batches = _chunk(np.argsort(arr, kind="stable"), batch_size)
        yield from batches
        return
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    mega = 50 * batch_size
    batches = []
    for start in range(0, n, mega):
        block = order[start : start + mega]
        batches.extend(_chunk(block[np.argsort(arr[block], kind="stable")], batch_size))
    for pick in rng.permutation(len(batches)):
        yield batches[pick]


def _chunk(order: np.ndarray, size: int) -> list[np.ndarray]:
    return [order[start : start + size] for start in range(0, len(order), size)]


def make_batches(
    rows: Sequence[Row], batch_size: int, *, seed: int, shuffle: bool
) -> Iterator[Batch]:
    for idx in bucket_batches([len(r.ids) for r in rows], batch_size, seed=seed, shuffle=shuffle):
        chosen = [rows[i] for i in idx]
        yield pad_batch([r.ids for r in chosen], [r.target for r in chosen], idx)


def predict_rows(
    rows: Sequence[Row], predict_batch: Callable[[Batch], np.ndarray], batch_size: int
) -> np.ndarray:
    """Predictions in the original row order (batches are length-sorted internally)."""
    out = np.zeros(len(rows), dtype=np.float64)
    for batch in make_batches(rows, batch_size, seed=0, shuffle=False):
        out[batch.index] = np.asarray(predict_batch(batch), dtype=np.float64).reshape(-1)
    return out
