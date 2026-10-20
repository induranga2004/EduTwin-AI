# edusync-blueprint-model

A **blueprint**, not a finished model. It shows a complete, reproducible way to turn
`sentence-transformers/all-MiniLM-L6-v2` into your own text classifier, shrink it with INT8 quantization, and
run it without PyTorch. Copy it, put your own labelled CSV in `data/`, and follow the steps.

Everything runs on the **CPU only** (no CUDA needed).

## Pipeline

```
 data/*.csv
     |
 download  -->  train  -->  evaluate (fp32)  -->  export  -->  quantize  -->  predict
 (base model)  (full       (test split)          (ONNX fp32)  (ONNX INT8,   (torch-free,
  ~92 MB       fine-tune)                                      + comparison   OnnxClassifier)
                                                               report)
```

Classifier: MiniLM encoder + mean pooling + dropout + linear head (`src/edusync_blueprint_model/model.py`).
Training uses PyTorch; inference and quantization use ONNX Runtime only.

## Quickstart (6 commands)

Run everything from the `edusync/` directory.

```bash
# 1. one-time install of the heavy extras (keeps the other workspace packages)
uv sync --inexact --package edusync-blueprint-model --extra finetune

# 2-6. the pipeline (uses the fake data/sample.csv until you change the config)
uv run --no-sync python -m edusync_blueprint_model download
uv run --no-sync python -m edusync_blueprint_model train
uv run --no-sync python -m edusync_blueprint_model export
uv run --no-sync python -m edusync_blueprint_model quantize
uv run --no-sync python -m edusync_blueprint_model predict "your text here"
```

Use `uv run --no-sync` (not plain `uv run`): a plain `uv run` re-syncs the environment and removes the extras.
`python -m edusync_blueprint_model info` shows which extras are installed and which stages are done.

## Folder tree

```
packages/blueprint_model/
  README.md                this file
  pyproject.toml           package and extras
  config/blueprint.toml    all settings (copy it for your own project)
  data/                    your CSV goes here (see data/README.md); sample.csv is FAKE
  artifacts/               generated models and reports; model files are in Git LFS (see artifacts/README.md)
  docs/
    quantization.md        quantization explained with steps and measured numbers
    customizing.md         how to adapt the blueprint to your own classifier
  src/edusync_blueprint_model/
    cli.py  __main__.py    command line interface
    config.py              typed config loading (pydantic)
    dataset.py             CSV loading, validation, stratified split
    download.py            fetch the pinned base model
    model.py               encoder + pooling + head
    train.py               CPU fine-tuning loop
    metrics.py             accuracy, macro-F1, per-class report
    onnx_export.py         ONNX export + parity check + sidecar files
    quantize.py            dynamic INT8 quantization + fp32 vs int8 report
    predict.py             torch-free OnnxClassifier
    deps.py                lazy imports with "install the X extra" errors
  tests/                   unit tests (fakes) and one `ml` end-to-end test
```

## CLI reference

Command form: `uv run --no-sync python -m edusync_blueprint_model [--config PATH] <command>`.

| Command | What it does | Needs extra |
|---|---|---|
| `info` | Show config, paths, installed extras, finished stages | none |
| `download` | Download the pinned base model into `artifacts/base/` | `download` |
| `train` | Full fine-tuning on CPU; writes `artifacts/finetuned/` and `reports/train_metrics.json` | `finetune` |
| `evaluate --model fp32\|int8` | Score an exported model on the test split; writes `reports/evaluation_<model>.json` (default `fp32`) | `onnx` |
| `export` | Export the fine-tuned model to ONNX fp32 and check it matches torch | `finetune` |
| `quantize` | Dynamic INT8 quantization, prints the fp32 vs int8 table, writes `reports/quantization_report.json` | `onnx` |
| `predict [--model fp32\|int8] TEXT... \| --file path` | Classify texts (default model `int8`); `--file` has one text per line | `onnx` |

Exit codes: 0 ok, 2 config/data/value error, 3 missing extra, 4 missing file (with a "run the X step first" hint).

## Extras

| Extra | Contains | Use it for |
|---|---|---|
| (none) | pydantic, numpy | `info` only |
| `download` | huggingface_hub | `download` |
| `onnx` | onnx, onnxruntime, tokenizers | torch-free inference and quantization (a deployment machine only needs this) |
| `finetune` | everything above plus torch (CPU build) and transformers | the whole pipeline |

## Requirements and limits

- CPU only. torch comes from the CPU-only PyTorch index configured in `edusync/pyproject.toml`.
- Python 3.12 (`requires-python = ">=3.12,<3.13"`).
- Base model download: about 92 MB (9 files).
- Disk use of `artifacts/` for the sample run: about 300 MB (base 88M, finetuned 88M, onnx 87M, quantized 23M).
- Full fine-tuning updates all weights; real datasets scale roughly linearly with rows and epochs.

## Documentation

- [docs/quantization.md](docs/quantization.md) - what quantization is, the steps, how to read the report, measured numbers.
- [docs/customizing.md](docs/customizing.md) - turn the blueprint into your own classifier.
- [data/README.md](data/README.md) - dataset format and rules.
- [artifacts/README.md](artifacts/README.md) - what each output folder contains.

## Status / unverified

- Everything was tested on one machine: Windows 11 + Python 3.12. macOS and Linux were not tested here.
- Accuracy numbers on `data/sample.csv` are meaningless (fake data, 9 test rows). Measure on your own data.
- The ONNX export uses the legacy TorchScript exporter. A dynamo-exporter fallback exists in the code but is untested.
- Only post-training dynamic INT8 is implemented. Static INT8, fp16 and quantization-aware training are not.
- Latency numbers vary between runs and machines; re-measure on your deployment hardware.
