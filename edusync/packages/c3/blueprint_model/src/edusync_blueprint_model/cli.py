"""Command line interface: info, download, train, evaluate, export, quantize, predict."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from edusync_blueprint_model.config import BlueprintConfig, ConfigError, load_config
from edusync_blueprint_model.dataset import load_splits
from edusync_blueprint_model.deps import MissingExtraError, install_command, is_available
from edusync_blueprint_model.download import download_base
from edusync_blueprint_model.metrics import summarize
from edusync_blueprint_model.onnx_export import FP32_NAME, export_onnx
from edusync_blueprint_model.predict import OnnxClassifier
from edusync_blueprint_model.quantize import (
    INT8_NAME,
    compare_models,
    format_report,
    quantize_model,
    write_report,
)
from edusync_blueprint_model.train import train as run_train

OPTIONAL_MODULES = ["huggingface_hub", "torch", "transformers", "onnx", "onnxruntime", "tokenizers"]
NEXT_STEP = {
    "train": "download",
    "export": "train",
    "quantize": "export",
    "evaluate": "export (and quantize for --model int8)",
    "predict": "export (and quantize for --model int8)",
}


def _model_dir(config: BlueprintConfig, kind: str) -> Path:
    return config.artifacts.onnx_dir if kind == "fp32" else config.artifacts.quantized_dir


def _load_classifier(config: BlueprintConfig, kind: str) -> OnnxClassifier:
    return OnnxClassifier.from_dir(_model_dir(config, kind))


def cmd_info(config: BlueprintConfig, args: argparse.Namespace) -> int:
    art = config.artifacts
    print(f"config folder : {config.config_dir}")
    print(f"base model    : {config.model.name} @ {config.model.revision[:10]}")
    print(f"max_length    : {config.model.max_length}")
    print(f"dataset       : {config.data.path}")
    print(f"artifacts dir : {art.root}")
    print("optional modules:")
    for name in OPTIONAL_MODULES:
        print(f"  {name:<16}{'installed' if is_available(name) else 'missing'}")
    if not all(is_available(m) for m in OPTIONAL_MODULES):
        print(f"  install everything with: {install_command('finetune')}")
    base = art.base_model_dir(config.model.name)
    stages = [
        ("download", base / "model.safetensors"),
        ("train", art.finetuned_dir / "head.pt"),
        ("export", art.onnx_dir / FP32_NAME),
        ("quantize", art.quantized_dir / INT8_NAME),
    ]
    print("pipeline stages:")
    for name, marker in stages:
        print(f"  {name:<10}{'done' if marker.is_file() else 'not run'}  ({marker.parent})")
    return 0


def cmd_download(config: BlueprintConfig, args: argparse.Namespace) -> int:
    print(f"Base model ready: {download_base(config)}")
    return 0


def cmd_train(config: BlueprintConfig, args: argparse.Namespace) -> int:
    result = run_train(config)
    for row in result.history:
        print(
            f"epoch {row['epoch']}: loss {row['train_loss']:.4f}  "
            f"val acc {row['val_accuracy']:.4f}  val macro-F1 {row['val_macro_f1']:.4f}"
        )
    print(f"best epoch {result.best_epoch} (val macro-F1 {result.best_val_macro_f1:.4f})")
    print(
        f"test accuracy {result.test['accuracy']:.4f}  test macro-F1 {result.test['macro_f1']:.4f}"
        f"  ({result.train_seconds:.1f}s)"
    )
    print(f"Saved to {result.finetuned_dir}")
    return 0


def _logits_fn(clf: OnnxClassifier) -> Any:
    return clf.logits


def cmd_evaluate(config: BlueprintConfig, args: argparse.Namespace) -> int:
    clf = _load_classifier(config, args.model)
    _, splits = load_splits(config)
    ids = {label: i for i, label in enumerate(clf.labels)}
    y_true = np.array([ids[e.label] for e in splits.test])
    y_pred = clf.logits([e.text for e in splits.test]).argmax(axis=-1)
    metrics = summarize(y_true, y_pred, clf.labels)
    reports = config.artifacts.reports_dir
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"evaluation_{args.model}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"{args.model} on {metrics['num_examples']} test examples")
    print(f"accuracy {metrics['accuracy']:.4f}  macro-F1 {metrics['macro_f1']:.4f}")
    for label, row in metrics["per_class"].items():
        print(
            f"  {label:<16} P {row['precision']:.3f}  R {row['recall']:.3f}  "
            f"F1 {row['f1']:.3f}  n={row['support']}"
        )
    return 0


def cmd_export(config: BlueprintConfig, args: argparse.Namespace) -> int:
    print(f"Exported: {export_onnx(config)}")
    return 0


def cmd_quantize(config: BlueprintConfig, args: argparse.Namespace) -> int:
    int8_path = quantize_model(config)
    print(f"Quantized: {int8_path}")
    fp32, int8 = _load_classifier(config, "fp32"), _load_classifier(config, "int8")
    _, splits = load_splits(config)
    ids = {label: i for i, label in enumerate(fp32.labels)}
    report = compare_models(
        _logits_fn(fp32),
        _logits_fn(int8),
        [e.text for e in splits.test],
        np.array([ids[e.label] for e in splits.test]),
        fp32.labels,
        fp32_path=config.artifacts.onnx_dir / FP32_NAME,
        int8_path=int8_path,
        runs=config.quantize.benchmark_runs,
    )
    print(format_report(report))
    print(f"Report: {write_report(report, config.artifacts.reports_dir)}")
    return 0


def cmd_predict(config: BlueprintConfig, args: argparse.Namespace) -> int:
    texts = list(args.texts)
    if args.file:
        lines = Path(args.file).read_text(encoding="utf-8").splitlines()
        texts.extend(line.strip() for line in lines if line.strip())
    if not texts:
        print("error: give some TEXT arguments or --file", file=sys.stderr)
        return 2
    for text, pred in zip(texts, _load_classifier(config, args.model).predict(texts), strict=True):
        print(f"{pred.label}\t{pred.score:.3f}\t{text}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edusync-blueprint", description=__doc__)
    parser.add_argument("--config", help="path to a blueprint.toml (default: packaged config)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, helptext in [
        ("info", "show config, paths, installed extras and finished stages"),
        ("download", "download the base model from Hugging Face"),
        ("train", "fine-tune on CPU"),
        ("export", "export the fine-tuned model to ONNX (fp32)"),
        ("quantize", "quantize to INT8 and compare with fp32"),
    ]:
        sub.add_parser(name, help=helptext)
    for name, helptext in [("evaluate", "score an ONNX model on the test split"), ("predict", "")]:
        p = sub.add_parser(name, help=helptext or "classify texts")
        p.add_argument(
            "--model", choices=["fp32", "int8"], default="int8" if name == "predict" else "fp32"
        )
        if name == "predict":
            p.add_argument("texts", nargs="*", metavar="TEXT")
            p.add_argument("--file", help="text file with one text per line")
    return parser


HANDLERS = {
    "info": cmd_info,
    "download": cmd_download,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "export": cmd_export,
    "quantize": cmd_quantize,
    "predict": cmd_predict,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("edusync_blueprint_model").setLevel(logging.INFO)
    try:
        config = load_config(args.config)
        return HANDLERS[args.command](config, args)
    except MissingExtraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        step = NEXT_STEP.get(args.command)
        if step:
            print(f"hint: run the '{step}' step first (edusync-blueprint {step.split()[0]})")
        return 4
    except (ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
