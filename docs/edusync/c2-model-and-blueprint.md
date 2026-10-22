# C2 difficulty model: interface, artifact spec and blueprint reuse

Companion to [c2-difficulty.md](c2-difficulty.md). Code: `edusync/packages/c2/src/edusync_c2/difficulty/models.py`
(model interface, baseline, artifact validation), `registry.py` (feature order and schema hash), `schemas.py`
(`FusionInput`, `ModelOutput`). Contract tests: `edusync/packages/c2/tests/difficulty/test_model_contract.py` and
`test_model_artifacts.py`.

## 1. Purpose and status

- **Phase 1 (now):** the only working model is `RuleBaselineModel`, a transparent, untrained rule. Nothing here is a
  trained model and no accuracy is claimed.
- **Phase 3 (later):** a MiniLM text encoder fused with the structured features, trained on lecturer labels, exported
  as an artifact folder and selected with `EDUSYNC_C2_MODEL=minilm`.
- What Phase 1 already delivers for Phase 3: the model interface, the artifact folder spec with validation
  (`OnnxFusionModel.from_artifacts`, no onnxruntime or torch needed), the feature schema hash and the contract test
  suite. ONNX inference itself is not implemented: `OnnxFusionModel.predict` raises `NotImplementedError`.

## 2. Model interface contract

```python
class DifficultyModel(Protocol):
    info: ModelInfo
    def predict(self, inputs: list[FusionInput]) -> list[ModelOutput | None]: ...
```

- **Input** `FusionInput`: `concept_id`, `text` (concept label, description and slide text, built by
  `fusion.build_fusion_input`) and `features`, one `FusionFeature` per registry entry in registry order
  (`registry.FEATURE_NAMES`, 24 names), each with `value`, `status` and a `missing` flag (unavailable features, and
  mocked ones by default, are missing with `value = None`).
- **Output** one entry per input, in order: a `ModelOutput` (`label`, optional `score`, optional `probabilities`,
  `explanation`) or `None` when that concept cannot be predicted.

Rules, enforced by `assert_model_contract` in `test_model_contract.py`:

1. The output list has the same length as the input list; `predict([])` returns `[]`.
2. Every non-`None` output has a `label` in `DifficultyLabel` (`Low`, `Medium`, `High`).
3. If `probabilities` is not `None` it uses exactly those three labels, none negative, and sums to 1 within 1e-6.
4. If `score` is present it is in [0, 1].
5. Deterministic: two calls with the same input return equal outputs.
6. Missing, mocked or unavailable features never make `predict` raise; the model returns a degraded result or `None`.
7. No invented confidence: `ModelOutput` has no `confidence` field and the contract rejects one. The API field
   `ConceptPrediction.confidence` is always `None` in Phase 1.

Why probabilities and confidence are never invented: the baseline is not trained, so a number shaped like a
probability would be fabricated. `probabilities` is `None` for the baseline; the baseline `score` is a raw weighted
mean, documented as "not a probability". A trained model may fill `probabilities` only after it has been evaluated;
calibrated confidence is a separate TODO (`registry.PHASE1_TODO`).

Label order is ordinal: **Low < Medium < High**. This is the order of `DifficultyLabel` and of `LABEL_ORDER` in
`models.py`. It is not alphabetical (alphabetical would give High < Low < Medium).

## 3. Model selection

`models.create_model(name, model_dir)` is the factory.

| `EDUSYNC_C2_MODEL` | `EDUSYNC_C2_MODEL_DIR` | Result |
|---|---|---|
| `baseline` (default) | ignored | `RuleBaselineModel` |
| `minilm` | unset | `ModelNotAvailableError("EDUSYNC_C2_MODEL=minilm needs EDUSYNC_C2_MODEL_DIR (Phase 3 artifacts)")` |
| `minilm` | set | `OnnxFusionModel.from_artifacts(dir)`; `ModelArtifactError` with a precise message if the folder breaks the spec |
| anything else | any | `ModelNotAvailableError` listing the valid names (`baseline, minilm`) |

Intended HTTP behaviour (implemented in the router, owned by the pipeline/router work, not in `models.py`):
`ModelNotAvailableError` and `ModelArtifactError` map to HTTP 501 on the difficulty route, and
`GET /api/c2/difficulty/info` reports `selected_model` (the env value) and `feature_schema_hash`
(`registry.feature_schema_hash()`), as declared on `DifficultyInfo` in `schemas.py`. Unverified here: the router wiring
was being edited concurrently when this document was written; check `router.py` before relying on the 501 mapping.

Note that with `minilm` and a valid folder the model still cannot predict in Phase 1, because inference is a Phase 3
task.

## 4. Artifact spec and Phase 3 hand-off

`models.ARTIFACT_SPEC` holds the same text as this table; `OnnxFusionModel.from_artifacts` enforces it.

| File | Check |
|---|---|
| folder | exists and is a directory |
| `model_int8.onnx` or `model_fp32.onnx` | at least one present; if both, int8 is used |
| `labels.json` | JSON list exactly `["Low", "Medium", "High"]` in that order (a bare list, not `{"labels": [...]}`) |
| `feature_schema.json` | object with `schema_version` (non-empty string); `names` equal to `FEATURE_NAMES` in order; `hash` equal to `feature_schema_hash()`. A mismatch says the registry changed since training |
| `inference_config.json` | object with `max_length` (int, 1..512, not a bool), `base_model` and `base_revision` (non-empty strings); optional `trained_on` (string) becomes `ModelInfo.trained_on` |
| `tokenizer.json` | present and non-empty |
| any JSON file | must parse; otherwise `<file> is not readable valid JSON` |

`feature_schema_hash()` is the sha256 of `name|group` lines in registry order. Adding, removing, renaming or
reordering a feature, or moving it to another group, changes the hash and invalidates every trained artifact.

Phase 3 hand-off checklist: the trained package must export
1. the ONNX graph (int8 and/or fp32) whose inputs include the feature vector in registry order;
2. `labels.json` as a bare ordinal list (the blueprint writes `{"labels": [...]}` sorted alphabetically, so this
   must change);
3. `feature_schema.json` written from `FEATURE_NAMES` and `feature_schema_hash()` at training time, plus the
   train-only imputation and scaling statistics (these may live inside the graph or in an extra file; the spec does
   not constrain extra files);
4. `inference_config.json` with `max_length`, `base_model`, `base_revision` and ideally `trained_on`;
5. `tokenizer.json`;
6. evaluation results (see section 5.4) kept next to the artifacts for the report, not required by the loader.

## 5. Blueprint reuse assessment

Reviewed: `edusync/packages/blueprint_model` (`src/edusync_blueprint_model/*.py`, `config/blueprint.toml`,
`docs/quantization.md`, `docs/customizing.md`, `README.md`). Line numbers below refer to the files as read.

### 5.1 What the blueprint supports

- CPU full fine-tuning of `sentence-transformers/all-MiniLM-L6-v2` with a pinned revision (`config/blueprint.toml`
  `[model]`; `train.py`).
- Mean pooling over real tokens, dropout, one linear head: `model.py:29-35` (head at `model.py:27`).
- ONNX export with a torch-vs-ONNX parity check, tolerance 1e-3 (`onnx_export.py`, `check_parity`), dynamic axes for
  batch and sequence.
- Dynamic INT8 quantization with ONNX Runtime (`quantize.py:194`, `quantize_dynamic`; `op_types` default
  `["MatMul", "Gather"]` in the shipped TOML) and an fp32 vs int8 comparison report; output name `model_int8.onnx`
  (`quantize.py:20`).
- Torch-free inference: `OnnxClassifier` in `predict.py` (uses `tokenizers` and `onnxruntime` only).
- TOML configuration (`config.py`), a CLI (`cli.py`), and the workspace coverage gate `fail_under = 90`
  (`edusync/pyproject.toml:76`).
- Sidecars it already writes: `tokenizer.json`, `labels.json`, `inference_config.json` (`onnx_export.py` `SIDECARS`),
  so the C2 artifact names deliberately match.

### 5.2 What is hard-wired to text only

| Where | Constraint |
|---|---|
| `dataset.py:15` | `Example(text, label)`; no feature columns |
| `dataset.py:24` | `LabelMap` sorts labels alphabetically, so ids give High < Low < Medium; wrong for ordinal difficulty |
| `dataset.py:83-109` | stratified random row split (`rng.shuffle(items)` per label); no group key, so slides/concepts of one lecture leak across train/val/test |
| `model.py:29` | `forward(input_ids, attention_mask)` only |
| `onnx_export.py:56` | ONNX `input_names` are only `input_ids` and `attention_mask` |
| `predict.py:59` | `OnnxClassifier.logits(texts)` takes plain strings and feeds only the two token tensors; `encode_batch(texts)` has no pair encoding |
| `config.py:18` | `extra="forbid"` on every section, so any new config key is rejected until the model is extended |
| `train.py:110-125` | optimizer groups are chosen by parameter-name prefix `encoder.` or `head.`; a fusion head must be named `head` (or the grouping changed), otherwise its parameters get no optimizer group |
| `onnx_export.py:40`, `dataset.py:40` | `labels.json` is `{"labels": [...]}` |
| `train.py` selection metric | best epoch chosen by validation macro-F1; no ordinal metric |

### 5.3 What a feature-fusion extension needs

- Config: optional `feature_columns`, `group_column`, `label_order` (explicit ordinal order replacing the alphabetical
  sort).
- Train-only `feature_stats.json`: per-feature impute value, mean/scale (fit on the train split only) and a missing
  mask per row, mirroring `fusion.to_vector` (value 0.0 plus mask 1 when missing).
- Fusion head: concatenate the pooled embedding with the scaled features and mask, then the linear (or small MLP)
  head, still named `head`.
- ONNX: a second float32 input `features` next to `input_ids` and `attention_mask`; extend the parity check and
  `OnnxClassifier` to feed it.
- Quantization caveat: dynamic INT8 on the small numeric head or the feature path may hurt accuracy; restrict
  `op_types` or keep the head in fp32 and compare fp32 vs int8 on the held-out set before shipping int8.
- Text input: encode concept and material as a `[SEP]` pair with `only_second` truncation, `max_length` 256 (the
  blueprint TOML says the model supports up to 256; the config validator allows up to 512). Latency caveat: longer
  sequences cost CPU time; the blueprint benchmarks only its default 128-token setup, so re-measure.

### 5.4 Research gaps the blueprint lacks

- Group split by lecture (no leakage between slides of one lecture).
- Balanced accuracy and ordinal metrics: quadratic weighted kappa (QWK) and MAE over the ordinal labels; the
  blueprint reports accuracy and macro-F1 (`metrics.py`).
- Baselines to beat: the rule baseline, majority class, and a feature-only model.
- Calibration of probabilities before any confidence is exposed.
- Small-data risk: lecturer labels will be few; full fine-tuning of all weights (`train.py`) overfits easily. Consider
  a frozen encoder by default (blueprint has no freeze option; none found with a search for `requires_grad`) with only
  the head trained, and unfreezing only if validation supports it.

### 5.5 Decision

Copy `blueprint_model` into a new package `packages/c2_difficulty_model` in Phase 3 and adapt it there. Accepted cost:
duplicated code and drift from the blueprint. Alternative considered: extend the blueprint in place; rejected because
the changes (ordinal labels, groups, second ONNX input, config keys) would change a deliberately generic blueprint
for every user of it.

What Phase 1 already provides for that work: the `DifficultyModel` interface, the artifact spec and its validator,
the feature schema hash, `fusion.to_vector` for the value/mask layout, and the contract suite that the trained model
must pass.

## 6. Open questions and unverified items

- HTTP 501 mapping and the `/info` fields are described from the design; confirm against `router.py`.
- Whether normalisation statistics live inside the ONNX graph or in an extra file is undecided.
- Real latency of a 256-token pair input on the target CPU is not measured.
- Whether int8 quantization is acceptable for the fusion head is unmeasured.
- The minimum number of lecturer labels needed for the fusion model to beat the rule baseline is unknown.
- `ModelKind.TODO` is reported by `OnnxFusionModel` because it cannot predict yet; Phase 3 should switch it to
  `ModelKind.TRAINED`.
- The blueprint line numbers will drift as that package changes.
