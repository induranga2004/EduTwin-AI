"""End-to-end T1 run: train, export, parity check, quantize, compare, report.

``impl`` is the torch layer (``edusync_c2_model.t1.torch_impl`` in the container, a fake in CI).
"""

import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from edusync_c2_model.public_data.clear import load_clear
from edusync_c2_model.public_data.licences import SOURCES
from edusync_c2_model.public_data.onestop import load_onestop
from edusync_c2_model.public_data.schema import TextExample
from edusync_c2_model.public_data.splits import partition, read_split
from edusync_c2_model.report import render_model_card, render_t1_report
from edusync_c2_model.t1.config import T1Config
from edusync_c2_model.t1.data import build_rows
from edusync_c2_model.t1.export import (
    FP32_NAME,
    INT8_NAME,
    check_io_contract,
    inference_config,
    make_predictor,
    parity_check,
    write_sidecars,
)
from edusync_c2_model.t1.quantize import compare_models
from edusync_c2_model.t1.trainer import fit

LOADERS = {"clear": (load_clear, True), "onestop": (load_onestop, False)}
TOKENIZER_NAME = "tokenizer.json"


class DataMissingError(FileNotFoundError):
    """A public data file or its sealed split is absent."""


def load_sources(data_dir: Path) -> tuple[dict[str, dict[str, list[TextExample]]], dict[str, bool]]:
    """Partitioned examples per source plus the ``higher_is_easier`` flags, using sealed splits."""
    parts: dict[str, dict[str, list[TextExample]]] = {}
    for source in SOURCES:
        data_file = Path(data_dir) / source.filename
        split_file = Path(data_dir) / "splits" / f"{source.key}_split.json"
        if not data_file.is_file() or not split_file.is_file():
            raise DataMissingError(f"need {data_file} and the sealed split {split_file}")
        parts[source.key] = partition(LOADERS[source.key][0](data_file), read_split(split_file))
    return parts, {key: LOADERS[key][1] for key in parts}


def run(
    config: T1Config,
    parts_by_source: dict[str, dict[str, list[TextExample]]],
    higher_is_easier: dict[str, bool],
    *,
    base_dir: Path,
    out_dir: Path,
    impl: ModuleType | Any,
    n_boot: int = 1000,
    log: Callable[[str], None] = print,
) -> dict:
    """Run everything and return the summary dict (also written to ``out_dir``)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    impl.set_threads(config.threads)
    impl.seed_everything(config.seed)
    tokenizer_json = Path(base_dir) / TOKENIZER_NAME
    encode = impl.load_encoder(tokenizer_json, config.max_length)
    rows, scalers = build_rows(
        parts_by_source, higher_is_easier, encode, limit_rows=config.limit_rows
    )
    log(f"rows: { {part: len(r) for part, r in rows.items()} }")

    model = impl.build_model(Path(base_dir), config.dropout)
    result = fit(impl.TorchBackend(model, config), rows, config, n_boot=n_boot, log=log)

    fp32, int8 = out_dir / FP32_NAME, out_dir / INT8_NAME
    exporter = impl.export_onnx(model, encode, fp32)
    run_fp32, inputs, outputs = impl.open_session(fp32, config.threads)
    check_io_contract(inputs, outputs)
    fp32_predict = make_predictor(encode, run_fp32)
    parity = parity_check(impl.torch_predictor(model, encode), fp32_predict)
    write_sidecars(out_dir, tokenizer_json, config, scalers)

    removed = impl.quantize_int8(fp32, int8, out_dir)
    run_int8, inputs8, outputs8 = impl.open_session(int8, config.threads)
    check_io_contract(inputs8, outputs8)
    eval_sets = {}
    for source in sorted(parts_by_source):
        subset = [r for r in rows["test"] if r.source == source]
        eval_sets[f"{source} test"] = (
            [r.text for r in subset],
            np.array([r.target for r in subset]),
            [r.group_id for r in subset],
        )
    comparison = compare_models(
        fp32_predict,
        make_predictor(encode, run_int8),
        eval_sets,
        fp32_path=fp32,
        int8_path=int8,
        threshold=config.ship_threshold,
    )
    summary = {
        "config": config.to_dict(),
        "n_train": len(rows["train"]),
        "best_epoch": result["best_epoch"],
        "history": result["history"],
        "metrics": result["metrics"],
        "quantization": comparison,
        "exporter": exporter,
        "parity_max_abs_diff": parity,
        "removed_temp_files": removed,
        "scalers": scalers,
        "versions": impl.versions(),
    }
    info = inference_config(config, scalers)
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_t1_report(summary), encoding="utf-8", newline="\n")
    (out_dir / "model_card.md").write_text(
        render_model_card(summary, info), encoding="utf-8", newline="\n"
    )
    return summary
