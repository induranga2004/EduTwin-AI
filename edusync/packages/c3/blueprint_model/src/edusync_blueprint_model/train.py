"""CPU fine-tuning loop (plain PyTorch) plus small unit-testable helpers."""

import copy
import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from edusync_blueprint_model.config import BlueprintConfig
from edusync_blueprint_model.dataset import Example, load_splits
from edusync_blueprint_model.metrics import selection_score, summarize
from edusync_blueprint_model.predict import decide, softmax

logger = logging.getLogger(__name__)


def linear_warmup_decay(step: int, total_steps: int, warmup_steps: int) -> float:
    """LR multiplier: linear 0 -> 1 over warmup, then linear 1 -> 0 at ``total_steps``."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / warmup_steps
    remaining = max(1, total_steps - warmup_steps)
    return max(0.0, (total_steps - step) / remaining)


def iter_batches(
    n: int, batch_size: int, *, seed: int = 0, shuffle: bool = True
) -> Iterator[np.ndarray]:
    """Yield index arrays covering ``range(n)`` in batches of ``batch_size``."""
    order = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


class TrainResult(BaseModel):
    best_epoch: int
    best_val_macro_f1: float
    selected_by: str = "macro_f1"
    test: dict[str, Any]
    history: list[dict[str, Any]]
    finetuned_dir: Path
    train_seconds: float = 0.0


def _no_decay(name: str) -> bool:
    return name.endswith(".bias") or "LayerNorm" in name


def _encode(tokenizer: Any, examples: list[Example], max_length: int) -> Any:  # pragma: no cover
    return tokenizer(
        [e.text for e in examples],
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )


def _predict(  # pragma: no cover - needs torch
    model: Any, tokenizer: Any, examples: list[Example], cfg: BlueprintConfig, labels: list[str]
) -> Any:
    """Predictions for ``examples`` (eval mode, no grad): argmax, or the [decision] threshold rule."""
    import torch

    model.eval()
    rows = []
    with torch.no_grad():
        for idx in iter_batches(len(examples), cfg.train.batch_size, shuffle=False):
            enc = _encode(tokenizer, [examples[i] for i in idx], cfg.model.max_length)
            logits = model(enc["input_ids"], enc["attention_mask"])
            rows.append(logits.numpy())
    d = cfg.decision
    return decide(softmax(np.concatenate(rows)), labels, d.positive_label, d.threshold)


def train(  # pragma: no cover - needs torch + weights
    config: BlueprintConfig, *, base_dir: Path | None = None
) -> TrainResult:
    """Fine-tune all weights on CPU, keep the best epoch by the configured validation score."""
    from edusync_blueprint_model.deps import require
    from edusync_blueprint_model.model import build_classifier, save_classifier

    torch = require("torch", "finetune")
    transformers = require("transformers", "finetune")
    cfg, tc = config, config.train
    base = Path(base_dir) if base_dir else cfg.artifacts.base_model_dir(cfg.model.name)
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"Base model not found in {base}; run the download step")

    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    if tc.threads > 0:
        torch.set_num_threads(tc.threads)

    labels, splits = load_splits(cfg)
    logger.info(
        "Data: %d train / %d val / %d test, %d classes",
        len(splits.train),
        len(splits.val),
        len(splits.test),
        len(labels),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(base))
    model = build_classifier(base, len(labels), tc.dropout)  # CPU only: never moved to a GPU

    groups = [
        (tc.encoder_lr, "encoder", True),
        (tc.encoder_lr, "encoder", False),
        (tc.head_lr, "head", True),
        (tc.head_lr, "head", False),
    ]
    param_groups = []
    for lr, part, decay in groups:
        params = [
            p
            for n, p in model.named_parameters()
            if n.startswith(part + ".") and (not _no_decay(n)) == decay
        ]
        wd = tc.weight_decay if decay else 0.0
        param_groups.append({"params": params, "lr": lr, "weight_decay": wd})
    optimizer = torch.optim.AdamW(param_groups)
    steps_per_epoch = -(-len(splits.train) // tc.batch_size)
    total = steps_per_epoch * tc.epochs
    warmup = int(total * tc.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: linear_warmup_decay(s, total, warmup)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    y_train = np.array([labels.id_of(e.label) for e in splits.train])
    y_val = np.array([labels.id_of(e.label) for e in splits.val])

    best = {"score": (-1.0, -1.0), "f1": -1.0, "epoch": 0, "state": None}
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, tc.epochs + 1):
        model.train()
        losses = []
        for idx in iter_batches(len(splits.train), tc.batch_size, seed=tc.seed + epoch):
            enc = _encode(tokenizer, [splits.train[i] for i in idx], cfg.model.max_length)
            logits = model(enc["input_ids"], enc["attention_mask"])
            loss = loss_fn(logits, torch.from_numpy(y_train[idx]))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_pred = _predict(model, tokenizer, splits.val, cfg, labels.labels)
        val = summarize(y_val, val_pred, labels.labels)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        if cfg.decision.positive_label is not None:
            row["val_positive_recall"] = val["per_class"][cfg.decision.positive_label]["recall"]
        history.append(row)
        logger.info("epoch %d: %s", epoch, row)
        score = selection_score(val, tc.select_by, cfg.decision.positive_label)
        if score > best["score"]:
            best = {
                "score": score,
                "f1": val["macro_f1"],
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }
    seconds = time.perf_counter() - started

    model.load_state_dict(best["state"])
    y_test = np.array([labels.id_of(e.label) for e in splits.test])
    test = summarize(y_test, _predict(model, tokenizer, splits.test, cfg, labels.labels), labels.labels)

    out = cfg.artifacts.finetuned_dir
    out.mkdir(parents=True, exist_ok=True)
    save_classifier(model, out, tc.dropout)
    tokenizer.save_pretrained(str(out / "tokenizer"))
    (out / "labels.json").write_text(labels.to_json(), encoding="utf-8")
    result = TrainResult(
        best_epoch=best["epoch"],
        best_val_macro_f1=best["f1"],
        selected_by=tc.select_by,
        test=test,
        history=history,
        finetuned_dir=out,
        train_seconds=seconds,
    )
    record = {
        "base_model": cfg.model.name,
        "base_revision": cfg.model.revision,
        "max_length": cfg.model.max_length,
        "hyperparameters": tc.model_dump(),
        "result": json.loads(result.model_dump_json()),
    }
    (out / "train_config.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    cfg.artifacts.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts.reports_dir / "train_metrics.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )
    return result
