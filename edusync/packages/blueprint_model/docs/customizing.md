# Customizing the blueprint for your own classifier

A recipe for turning this blueprint into a classifier for your own labels and data. Commands are run from the
`edusync/` directory. For the quantization step in detail see [quantization.md](quantization.md).

## Contents

1. [Copy and edit the config](#1-copy-and-edit-the-config)
2. [Prepare your data](#2-prepare-your-data)
3. [Run the pipeline](#3-run-the-pipeline)
4. [Iterate](#4-iterate)
5. [Customization points in code](#5-customization-points-in-code)
6. [Use the result in a service](#6-use-the-result-in-a-service-torch-free)
7. [Tests](#7-tests)
8. [Sharing models](#8-sharing-models)

## 1. Copy and edit the config

```bash
cp packages/blueprint_model/config/blueprint.toml packages/blueprint_model/config/my_project.toml
```

Relative paths inside the file (`[data] path`, `[paths] artifacts_dir`) are resolved against the folder that
contains the config file. Unknown keys are rejected, so typos are reported as errors (exit code 2).

To keep your outputs separate from the sample run, give your config its own `artifacts_dir`.

| Section | Field | Default | What it controls / when to change it |
|---|---|---|---|
| `[model]` | `name` | `sentence-transformers/all-MiniLM-L6-v2` | Base model on Hugging Face. Change only for a different base (see section 5). |
| `[model]` | `revision` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` | Pinned commit of the base repo, so everyone starts from the same weights. Change together with `name`. |
| `[model]` | `max_length` | `128` | Max tokens per text (1 token is roughly a word piece). Raise it (the base model supports up to 256) if your texts are long and get truncated; lower it for speed. Training and inference use the same value. |
| `[data]` | `path` | `../data/sample.csv` | Your CSV. Always change. |
| `[data]` | `text_column` / `label_column` | `text` / `label` | Column names in your CSV. |
| `[data]` | `val_fraction` | `0.2` | Share held out to choose the best epoch. |
| `[data]` | `test_fraction` | `0.2` | Share held out for the final evaluation and the quantization comparison. With little data, make sure the test split ends up with 100+ rows if you can. |
| `[data]` | `seed` | `42` | Split shuffling seed. Keep fixed to compare runs. |
| `[train]` | `epochs` | `4` | Passes over the training data. More if the validation score is still rising at the end; fewer if it peaks early. |
| `[train]` | `batch_size` | `16` | Rows per step. Larger is faster per epoch but uses more RAM. |
| `[train]` | `encoder_lr` | `3e-5` | Learning rate for the pretrained MiniLM layers. Lower (for example 1e-5) if training is unstable or forgets; higher if it barely learns. |
| `[train]` | `head_lr` | `1e-3` | Learning rate for the new linear layer (starts random, so it can be larger). |
| `[train]` | `weight_decay` | `0.01` | L2-style regularization (not applied to biases and LayerNorm). |
| `[train]` | `warmup_ratio` | `0.1` | Share of steps spent warming the learning rate up; it then decays linearly to 0. |
| `[train]` | `dropout` | `0.1` | Dropout before the head. Raise (0.2-0.3) if overfitting. |
| `[train]` | `threads` | `0` | PyTorch CPU threads; 0 = library default (all cores). Set it to leave cores free for other work. |
| `[train]` | `seed` | `42` | Training seed (initial head weights, batch order). |
| `[quantize]` | `weight_type` | `"qint8"` | `"qint8"` or `"quint8"`. |
| `[quantize]` | `per_channel` | `false` | Per-channel scales. |
| `[quantize]` | `op_types` | `["MatMul", "Gather"]` | Operators to quantize; keep `Gather` (embedding table). |
| `[quantize]` | `benchmark_runs` | `50` | Timed runs per batch size in the comparison. Raise for steadier latency numbers. |
| `[paths]` | `artifacts_dir` | `../artifacts` | Where all outputs go (git-ignored). |

Note: `val_fraction + test_fraction` must be below 1.0. The `[quantize]` fields are explained in
[quantization.md](quantization.md).

## 2. Prepare your data

Format and loader rules are in [../data/README.md](../data/README.md); do not repeat them here. Guidance beyond
those rules:

- **Labels.** Any strings; use consistent spelling and casing (`Relevant` and `relevant` are two classes).
  Classes are sorted alphabetically and mapped to ids automatically.
- **Minimum rows.** The loader enforces at least 3 rows per class and at least 2 classes. That only lets the
  pipeline run; it is far too little to learn anything useful.
- **How many is enough.** There is no fixed number and none is verified here. Qualitatively: tens of examples
  per class can give a rough result with a pretrained encoder, hundreds per class are more comfortable, and
  more rows help most for classes that look alike. Your test split needs enough rows (100+ overall, more for
  rare classes) for the metrics to mean anything.
- **Balance.** The split is stratified, so rare classes keep their share, but heavy imbalance can still hide a
  poor minority class behind a good accuracy. Look at macro-F1 and per-class recall.
- **Leakage.** The split is by row. Exact or near-duplicate texts that land in both train and test make the
  scores look better than reality. Remove duplicates before training (and group related rows, such as many
  sentences from the same document, so they are not split across train and test yourself, since the loader
  does not do this).
- **Do not commit private data.** `data/` is git-ignored except `sample.csv`.

## 3. Run the pipeline

Pass your config with `--config` (it goes before the command):

```bash
uv sync --inexact --package edusync-blueprint-model --extra finetune   # once
P="uv run --no-sync python -m edusync_blueprint_model --config packages/blueprint_model/config/my_project.toml"
$P info
$P download
$P train
$P evaluate --model fp32
$P export
$P quantize
$P evaluate --model int8
$P predict "an example text"
```

Note: `evaluate` needs the exported model, so run `export` before it (`train` already prints test metrics).
If a step reports exit code 4, it prints which earlier step to run. Exit code 3 means an extra is missing and
the message shows the install command.

## 4. Iterate

`train` prints per-epoch loss, validation accuracy and validation macro-F1, and the best epoch is chosen by
validation macro-F1. The same history is in `artifacts/reports/train_metrics.json` (`history`, `best_epoch`,
`best_val_macro_f1`, `test`). The hyperparameters, base revision and results are also stored in
`artifacts/finetuned/train_config.json`.

| What you see | Likely cause | Try first |
|---|---|---|
| Train loss keeps falling but validation macro-F1 stalls or drops | Overfitting | More or cleaner data; fewer epochs; higher `dropout`; lower `encoder_lr` |
| Train loss stays high and validation is poor | Underfitting or too little training | More `epochs`; check labels are consistent; raise `encoder_lr` slightly; check texts are not truncated (`max_length`) |
| Validation jumps around between epochs | Tiny validation set or learning rate too high | More data; lower `encoder_lr` |
| Great validation, poor test | Small splits or leakage between train and validation | Bigger splits; remove duplicates |
| One class has low recall | Imbalance or ambiguous labels | More examples of that class; review labels (class weighting is not built in, see section 5) |

Change one thing at a time and keep the same `seed` values so runs are comparable. Do not tune against the test
split; use validation for decisions and look at test once at the end.

## 5. Customization points in code

| You want to change | File | Notes |
|---|---|---|
| Head or pooling | `src/edusync_blueprint_model/model.py` | `MiniLMClassifier.forward` does mean pooling over real tokens, dropout, then one `nn.Linear`. Swap in CLS pooling, a bigger head, etc. |
| Loss, class weights, optimizer, schedule | `src/edusync_blueprint_model/train.py` | The loss is `torch.nn.CrossEntropyLoss()` (pass `weight=` for class weights); the optimizer is AdamW with separate learning rates for encoder and head; the schedule is `linear_warmup_decay`. |
| Tokenization / truncation | `[model] max_length`, `_encode` in `train.py`, and `OnnxClassifier._encode` in `predict.py` | Keep training and inference tokenization identical. Inference uses the saved `tokenizer.json`. |
| Extra metrics | `src/edusync_blueprint_model/metrics.py` | `summarize()` builds the dict used by `train` and `evaluate`; add fields there. |
| Different base model | `[model] name` and `revision` | See the caveats below. |

Caveats for a different base model:

- `model.py` assumes a BERT-style encoder: `transformers.AutoModel` output whose first element is the
  last hidden state (`hidden[0]`, shape batch x tokens x `encoder.config.hidden_size`) and inputs
  `input_ids` and `attention_mask`. Models that need `token_type_ids` or have a different output layout need
  code changes (and a matching change to the ONNX export inputs in `onnx_export.py`).
- `BASE_PATTERNS` in `download.py` lists the files to fetch (including `1_Pooling/config.json`, which only
  sentence-transformers repos have), and `REQUIRED_FILES` lists what must exist afterwards. They must match
  your repo's real file names, and the repo needs a `tokenizer.json` (a fast tokenizer).
- Pin a `revision` (commit hash) so the base stays reproducible.
- The quantization size and speed numbers in [quantization.md](quantization.md) belong to MiniLM-L6-v2;
  re-measure for another model. Nothing here has been tested with a base other than MiniLM-L6-v2.

## 6. Use the result in a service (torch-free)

The `onnx/` or `quantized/` folder is self-contained. Needs only the `onnx` extra, not torch.

```python
from edusync_blueprint_model.predict import OnnxClassifier

clf = OnnxClassifier.from_dir("artifacts/quantized")  # folder with model_int8.onnx + sidecars

preds = clf.predict(["first text", "second text"])  # list[Prediction]
for p in preds:
    print(p.label, p.score, p.probabilities)

logits = clf.logits(["first text"])  # numpy array, shape (1, num_labels)
```

`Prediction` has `label`, `score` (probability of the chosen label) and `probabilities` (all labels).

Sketch (not implemented here) of an EduSync component such as C3 (`edusync/packages/c3`, routes in
`router.py`) using it: build the classifier once per process and expose it through a FastAPI dependency, the
way C1 injects its transcriber (`get_transcriber` and `TranscriberDep = Annotated[Transcriber,
Depends(get_transcriber)]` in `packages/c1/src/edusync_c1/api.py` and `transcription.py`). A route would
declare a dependency, call `clf.predict(...)`, and tests would override the dependency with a fake, exactly as
the C1 tests do for the transcriber. Keep the model folder path in settings, and cache the loaded classifier
(for example with `functools.lru_cache`) so it is not reloaded per request.

## 7. Tests

Run from `edusync/`:

```bash
uv run --no-sync pytest packages/blueprint_model        # unit tests (the `ml` marker is deselected by default)
uv run --no-sync pytest packages/blueprint_model -m ml  # end-to-end test with real weights (slow, needs the finetune extra)
uv run --no-sync ruff check packages/blueprint_model
```

- Unit tests use fakes (fake downloader, fake ONNX session and tokenizer, injected timers) and a small TOML
  written by `tests/conftest.py`, so they need no torch, no network and no model files.
- Tests that need real ML models or weights carry `pytest.mark.ml` (see `tests/test_pipeline_e2e.py`) and are
  deselected by default through `addopts = "... -m 'not ml'"` in `edusync/pyproject.toml`.
- Heavy functions that need torch or onnxruntime are marked `# pragma: no cover`; keep new logic in small pure
  functions (like `linear_warmup_decay`, `split_dataset`, `compare_models`) so it is unit-testable.
- When you add logic, add a test next to the similar ones in `tests/`. The workspace enforces a
  **90% coverage gate** (`fail_under = 90` in `edusync/pyproject.toml`; run `pytest` with `--cov` to see it),
  and ruff uses line length 100.

## 8. Sharing models

Artifacts are git-ignored on purpose. Share them through team storage (a release asset, a drive folder, a
model registry), not the repository. Next to the model, record:

- the base model `name` and `revision` (already in `inference_config.json` and `train_config.json`),
- the dataset version (file name, date or hash; the data itself is not in git),
- a copy of the config file you used,
- the evaluation numbers, the exact commands, and the library versions you ran with.

See [../artifacts/README.md](../artifacts/README.md) for the folder layout.
