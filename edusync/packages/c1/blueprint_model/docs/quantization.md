# Quantization explained, with steps

This guide explains what INT8 quantization is, which kind this blueprint uses, and exactly how to run it and
judge the result. Commands are run from the `edusync/` directory.

## Contents

1. [What quantization is](#1-what-quantization-is)
2. [Kinds of quantization](#2-kinds-of-quantization)
3. [Step-by-step guide](#3-step-by-step-guide)
4. [How to read the comparison report](#4-how-to-read-the-comparison-report)
5. [Measured numbers](#5-measured-numbers)
6. [Tuning and pitfalls](#6-tuning-and-pitfalls)
7. [Where to go next](#7-where-to-go-next)

## 1. What quantization is

A trained model is mostly a large pile of numbers (weights). By default each one is a 32-bit float (fp32,
4 bytes). Quantization stores them as 8-bit integers (INT8, 1 byte) plus a little extra information to map
them back to real values.

The mapping uses a **scale** and a **zero point**:

```
q  = round(x / scale) + zero_point        (quantize: float -> int8)
x' = (q - zero_point) * scale             (dequantize: int8 -> approximate float)
```

Worked example, symmetric (zero_point = 0) with weights between -1.27 and 1.27. The int8 range is -127..127,
so `scale = 1.27 / 127 = 0.01`:

| x (fp32) | q = round(x / 0.01) | x' = q * 0.01 | error |
|---|---|---|---|
| 0.500 | 50 | 0.50 | 0.000 |
| -0.837 | -84 | -0.84 | 0.003 |
| 0.0042 | 0 | 0.00 | 0.0042 |

Values smaller than half a step (0.005 here) collapse to zero, and every value is off by at most half a step.
That rounding error is the price of quantization.

**Why do it**

- Smaller file: 1 byte instead of 4 per weight.
- Less RAM when the model is loaded.
- Faster on CPUs: integer math can use SIMD instructions (for example AVX2 or VNNI) that process more numbers
  per instruction.

**What it costs**

- Rounding error, so outputs change slightly. This can reduce accuracy a little, and for a poorly trained or
  borderline model it can flip predictions. How much you lose depends on your model and data: measure it.
- Some parts are more sensitive than others (embedding tables and LayerNorm are commonly cited as sensitive).
  This blueprint quantizes `MatMul` and `Gather` weights and leaves the rest (such as LayerNorm) in fp32.

## 2. Kinds of quantization

| Kind | How it works | Calibration data? | Typical use | In this blueprint |
|---|---|---|---|---|
| Post-training **dynamic** INT8 | Weights converted to int8 offline; activations quantized on the fly at run time | No | Transformers on CPU, easiest | **Implemented and tested** |
| Post-training **static** INT8 | Weights and activations both quantized offline using a calibration set | Yes | Often fastest on CPU; more setup | Not implemented |
| fp16 / bf16 | Half-precision floats; halves the size | No | Mostly GPUs and newer CPUs | Not implemented |
| PyTorch `torch.quantization.quantize_dynamic` | Same idea as dynamic INT8 but stays in PyTorch | No | Quick experiment without ONNX | Not implemented |
| Quantization-aware training (QAT) | Simulates quantization during training so the model learns to cope | Training data | Best accuracy, most work | Not implemented |
| INT4 / weight-only low-bit | 4-bit (or lower) weights, usually for large language models | Often | LLMs | Not implemented |

Only dynamic INT8 through ONNX Runtime is implemented and tested here. The other rows are options for you to
explore (see [section 7](#7-where-to-go-next)).

## 3. Step-by-step guide

### 3.1 Prerequisites and install

```bash
uv sync --inexact --package edusync-blueprint-model --extra finetune
uv run --no-sync python -m edusync_blueprint_model info
```

`--inexact` keeps the other workspace packages installed. Always run later commands with `uv run --no-sync`;
a plain `uv run` re-syncs and removes the extras. `info` should list all optional modules as installed.

### 3.2 Download and fine-tune

```bash
uv run --no-sync python -m edusync_blueprint_model download
uv run --no-sync python -m edusync_blueprint_model train
```

`train` prints loss and validation metrics per epoch and the test result, and writes `artifacts/finetuned/` and
`artifacts/reports/train_metrics.json`. Do not quantize a model you do not trust yet (see section 6).

### 3.3 Export to ONNX

```bash
uv run --no-sync python -m edusync_blueprint_model export
```

Files written to `artifacts/onnx/`: `model_fp32.onnx`, `tokenizer.json`, `labels.json`,
`inference_config.json`. The pooling and the classification head are inside the ONNX graph. The command also
checks that ONNX Runtime reproduces the torch logits (on a batch and on a single text) and fails if the
difference exceeds 1e-3. It uses the legacy TorchScript exporter, opset 17.

### 3.4 Evaluate fp32

Evaluation runs on the exported ONNX models, so it comes after the export:

```bash
uv run --no-sync python -m edusync_blueprint_model evaluate --model fp32
```

This is your reference: accuracy, macro-F1 and per-class precision/recall on the test split. It writes
`artifacts/reports/evaluation_fp32.json`.

### 3.5 Quantize

```bash
uv run --no-sync python -m edusync_blueprint_model quantize
```

What happens inside, in order (`quantize_model` in `src/edusync_blueprint_model/quantize.py`):

1. `onnxruntime.quantization.shape_inference.quant_pre_process(fp32, model_prep.onnx, skip_symbolic_shape=True)`
   - ONNX shape inference and graph optimization. If it fails, a warning is logged and the raw model is used.
2. `onnxruntime.quantization.quantize_dynamic(source, model_int8.onnx, weight_type=..., per_channel=..., op_types_to_quantize=...)`
   with the `[quantize]` values from the config.
3. The temporary `model_prep.onnx` is deleted.
4. `tokenizer.json`, `labels.json` and `inference_config.json` are copied next to the INT8 model in
   `artifacts/quantized/`.

Then it loads both models, compares them on the test split, prints the table and writes
`artifacts/reports/quantization_report.json`. The config fields (`[quantize]` in `config/blueprint.toml`):

| Field | Default | Meaning |
|---|---|---|
| `weight_type` | `"qint8"` | `"qint8"` (signed) or `"quint8"` (unsigned) weights |
| `per_channel` | `false` | one scale per output channel instead of one per tensor |
| `op_types` | `["MatMul", "Gather"]` | which operators get quantized weights |
| `benchmark_runs` | `50` | timed runs per batch size in the comparison |

### 3.6 Read the report

See [section 4](#4-how-to-read-the-comparison-report).

### 3.7 Evaluate INT8 on the test split

```bash
uv run --no-sync python -m edusync_blueprint_model evaluate --model int8
```

Writes `artifacts/reports/evaluation_int8.json`. Compare per-class recall with the fp32 run, not just the
overall accuracy. Example output on the fake sample data (meaningless numbers, shown only for the format):

```
int8 on 9 test examples
accuracy 0.3333  macro-F1 0.1667
  class_a          P 0.333  R 1.000  F1 0.500  n=3
  class_b          P 0.000  R 0.000  F1 0.000  n=3
  class_c          P 0.000  R 0.000  F1 0.000  n=3
```

### 3.8 Predict with INT8 (torch-free)

```bash
uv run --no-sync python -m edusync_blueprint_model predict --model int8 "some text to classify"
```

Prints `label<TAB>score<TAB>text` per input. `int8` is the default for `predict`. Use `--file texts.txt` for
one text per line. This path imports only ONNX Runtime and `tokenizers`, never torch.

### 3.9 Deploy

Copy the whole `artifacts/quantized/` folder (it is self-contained) to your service and load it:

```python
from edusync_blueprint_model.predict import OnnxClassifier

clf = OnnxClassifier.from_dir("path/to/quantized")
print(clf.predict(["some text to classify"]))
```

The deployment machine needs only the `onnx` extra. Measure latency on that machine.

## 4. How to read the comparison report

`quantize` prints a table (one row per metric, columns fp32 / int8 / change) and a line with the prediction
agreement.

| Row | Meaning | What good looks like |
|---|---|---|
| size (MB) | File sizes, in MiB (1 MiB = 1.049 MB) | Several times smaller (see section 5) |
| batch N p50 / p95 (ms) | Median and 95th-percentile latency for batch sizes 1 and 16, after warm-up | INT8 clearly faster; p95 close to p50 |
| accuracy | Fraction correct on the test split | Delta within your tolerance |
| macro F1 | F1 averaged over classes, so rare classes count equally | Delta within your tolerance |
| prediction agreement | Share of test texts where fp32 and INT8 predict the same class | High; the rest are flipped predictions |

Decision checklist:

- [ ] The fp32 model is good enough on its own (quantization cannot fix a weak model).
- [ ] The test set is large enough. The tool itself warns when there are fewer than 100 test examples; below
      that, differences are mostly noise. Prefer several hundred, and more if you have rare classes.
- [ ] Agreement is high.
- [ ] The accuracy and macro-F1 deltas are inside a tolerance you decided before looking (for example "at most
      one point lost").
- [ ] Per-class recall for rare classes has not dropped (`evaluate --model fp32` vs `--model int8`).
- [ ] Latency is better on the machine you will deploy to, not just on your laptop.
- [ ] You keep the fp32 `onnx/` folder as the reference and can switch back.

## 5. Measured numbers

Measured on one PC. Treat them as an example, not a guarantee.

Setup: AMD Ryzen 7 8840HS (8 cores / 16 threads), Windows 11 Pro, Python 3.12, torch 2.14.1+cpu,
transformers 5.20.2, onnx 1.23.2, onnxruntime 1.31.0, tokenizers 0.23.3. Inference: ONNX Runtime
`CPUExecutionProvider`.

**File size** (exported model): fp32 90.4 MB (decimal; the report prints 86.2 because it uses MiB) vs INT8
22.8 MB (21.8 MiB), which is 3.96x smaller. The base repo on Hugging Face also ships a prebuilt int8 ONNX of
about 23.0 MB, the same ballpark.

**Which operators are quantized** (`quantize_dynamic`, QInt8):

| `op_types` | per_channel | Size | Smaller by |
|---|---|---|---|
| `MatMul` only | false | 58.6 MB | 1.54x |
| `MatMul` + `Gather` | false | 22.9 MB | 3.95x |
| `MatMul` + `Gather` | true | 23.0 MB | 3.93x (no real size gain) |

The 30522 x 384 word-embedding table is a `Gather` operator and is roughly half the model. It stays fp32
unless `Gather` is in `op_types`. That is why the shipped config includes it.

**Latency** (median / p95 of 50 timed runs after warm-up; two separate runs, values vary run to run):

| Batch | fp32 p50 | INT8 p50 | Speed-up | fp32 p95 | INT8 p95 |
|---|---|---|---|---|---|
| 1 | 3.0-4.1 ms | 1.4-2.1 ms | about 2.0-2.1x | 3.8-6.8 ms | 1.7-3.7 ms |
| 16 | 21.8-23.9 ms | 11.1-12.2 ms | about 2.0x | 23.5-27.5 ms | 12.7-16.1 ms |

**Parity**: the exported fp32 ONNX matches torch logits with a max absolute difference of 3.05e-07
(checked on a batch and on a single text).

**Accuracy**: on the fake 45-row sample (9 test rows) fp32 accuracy was 0.667 and INT8 0.333, with 66.7%
prediction agreement. This is **not meaningful**: 9 rows, a barely trained model, near-tied logits that flip
easily. It only illustrates why you need a real test set. The real accuracy impact must be measured on your
own data; no typical loss figure is claimed here.

Other: the base download is about 92 MB (9 files, about 16 s here); fine-tuning 27 training rows for 4 epochs
took about 1.6 s. That is tiny; real datasets take longer, roughly in proportion to rows and epochs.

## 6. Tuning and pitfalls

- **Which operators.** Include `Gather` (the embedding table) or you only get about 1.5x smaller instead of
  about 4x. `MatMul` covers the attention and feed-forward weights.
- **per_channel.** One scale per output channel can preserve accuracy better for weights with very different
  ranges per channel. Here it gave no size gain (23.0 vs 22.9 MB); whether it helps accuracy for you is
  something to measure. Check latency too.
- **qint8 vs quint8.** Signed vs unsigned weight storage. `qint8` is the default; try `quint8` only if you have
  a reason (for example a specific kernel on your target hardware) and measure.
- **CPU instruction sets.** Quantized models may run noticeably faster on CPUs with AVX2/VNNI and less so on
  others. Always benchmark on the deployment machine.
- **`quant_pre_process` temp-file gotcha.** ONNX Runtime's full symbolic shape inference always fails on this
  model's dynamic sequence axis ("Incomplete symbolic shape inference") and leaves a model-sized temporary
  `*.data` file in the working directory on every failed attempt. The blueprint calls it with
  `skip_symbolic_shape=True` (shape inference and optimization still run). If you write your own script, do
  the same or clean up the stray files.
- **Dynamic axes.** The export marks batch and sequence length as dynamic, so any batch size and text length
  (up to `max_length`) works. Keep `max_length` in the config consistent between training and inference.
- **Tiny test sets.** Under 100 test rows, accuracy differences between fp32 and INT8 are noise. Do not
  conclude anything from them.
- **Quantize only a model you trust.** If fp32 is poor, INT8 will be poor. Validate fp32 first.
- **Reproducibility.** The base revision is pinned in the config, seeds are set (`data.seed`, `train.seed`),
  and `artifacts/finetuned/train_config.json` records the hyperparameters. Keep a copy of your config file
  next to the artifacts you share. Dynamic quantization needs no calibration data, so it does not depend on
  a random sample.

## 7. Where to go next

None of these are implemented in this blueprint; they are pointers only.

- **Static INT8 with ONNX Runtime.** `onnxruntime.quantization.quantize_static` with a calibration data
  reader that yields representative `input_ids` and `attention_mask` batches from your training data.
  Often faster, but needs calibration data and more care.
- **fp16 / bf16.** Convert weights to half precision (for example with the ONNX Runtime float16 tools).
  Mainly useful on GPUs or CPUs with native support.
- **Quantization-aware training.** Fine-tune with simulated quantization to recover accuracy if post-training
  quantization loses too much.
