"""The torch-free training loop. A ``Backend`` (``torch_impl.TorchBackend`` or a test fake) does the
numeric work; this module owns the schedule, epoch loop, best-epoch selection on VAL Spearman and
the single read of the test split at the very end."""

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

from edusync_c2_model.regression_metrics import evaluate_regressor, spearman
from edusync_c2_model.t1.config import T1Config
from edusync_c2_model.t1.data import Batch, Row, make_batches, predict_rows


class Backend(Protocol):
    def train_step(self, batch: Batch, lr_scale: float) -> float: ...

    def predict_batch(self, batch: Batch) -> np.ndarray: ...

    def snapshot(self) -> Any: ...

    def restore(self, state: Any) -> None: ...


def lr_scale(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear 0 -> 1 over warmup, then linear 1 -> 0 at ``total_steps``."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def evaluate_by_source(
    rows: Sequence[Row], preds: np.ndarray, seed: int, *, n_boot: int
) -> dict[str, dict]:
    """``evaluate_regressor`` per source ('clear', 'onestop') on z-scored difficulty."""
    out = {}
    for source in sorted({r.source for r in rows}):
        pick = [i for i, r in enumerate(rows) if r.source == source]
        out[source] = evaluate_regressor(
            [rows[i].target for i in pick],
            preds[pick],
            [rows[i].group_id for i in pick],
            seed,
            n_boot=n_boot,
        )
    return out


def mean_spearman(rows: Sequence[Row], preds: np.ndarray) -> float:
    """Mean of the per-source Spearman values (0.0 when none is defined)."""
    values = []
    for source in sorted({r.source for r in rows}):
        pick = [i for i, r in enumerate(rows) if r.source == source]
        value = spearman([rows[i].target for i in pick], preds[pick])
        if value is not None:
            values.append(value)
    return float(np.mean(values)) if values else 0.0


def fit(
    backend: Backend,
    rows: dict[str, list[Row]],
    config: T1Config,
    *,
    n_boot: int = 1000,
    log: Callable[[str], None] = lambda _msg: None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Train, keep the best epoch by VAL mean Spearman, then evaluate val and test once.

    Returns history, the best epoch and per-source metrics (z-scored difficulty).
    """
    train = rows["train"]
    steps_per_epoch = -(-len(train) // config.batch_size)
    total = steps_per_epoch * config.epochs
    warmup = int(config.warmup_ratio * total)
    step = 0
    best_score, best_epoch, best_state = -np.inf, 0, None
    history = []
    for epoch in range(1, config.epochs + 1):
        started = clock()
        losses = []
        for batch in make_batches(train, config.batch_size, seed=config.seed + epoch, shuffle=True):
            losses.append(backend.train_step(batch, lr_scale(step, total, warmup)))
            step += 1
        val_preds = predict_rows(rows["val"], backend.predict_batch, config.batch_size)
        score = mean_spearman(rows["val"], val_preds)
        entry = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_spearman": score,
            "seconds": clock() - started,
        }
        history.append(entry)
        log(f"epoch {epoch}: {entry}")
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, backend.snapshot()
    backend.restore(best_state)
    metrics = {}
    for part in ("val", "test"):  # the test split is read here, once, after model selection
        preds = predict_rows(rows[part], backend.predict_batch, config.batch_size)
        metrics[part] = evaluate_by_source(rows[part], preds, config.seed, n_boot=n_boot)
    return {"history": history, "best_epoch": best_epoch, "metrics": metrics}
