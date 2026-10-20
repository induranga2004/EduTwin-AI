# Artifacts

Everything the blueprint produces is written here. Model files (`*.onnx`, `*.pt`, `*.pth`, `*.safetensors`,
`*.joblib`) are committed through **Git LFS**; small files (JSON, tokenizer files, reports) are normal git.
The models committed with the blueprint were trained on the FAKE sample dataset: they only prove the pipeline
works and must not be used for anything real. See [Git LFS](#git-lfs) below.

```
artifacts/
  base/<model-name>/       base MiniLM-L6-v2 weights and tokenizer from Hugging Face (about 92 MB, pinned revision)
  finetuned/               your fine-tuned model
    encoder/               fine-tuned MiniLM weights (PyTorch format)
    tokenizer/             tokenizer files
    head.pt                the classification layer
    labels.json            class names in id order
    train_config.json      hyperparameters, base revision, metrics (for reproducibility)
  onnx/                    exported fp32 model for torch-free inference
    model_fp32.onnx
    tokenizer.json  labels.json  inference_config.json
  quantized/               INT8 model (about 4x smaller)
    model_int8.onnx
    tokenizer.json  labels.json  inference_config.json
  reports/
    train_metrics.json     validation/test metrics per epoch
    quantization_report.json   fp32 vs int8: size, latency, accuracy, agreement
```

Each stage reads the previous stage's folder, so you can delete a later folder and rebuild it.

## Using a model from another folder

`onnx/` and `quantized/` are self-contained. Copy either folder anywhere (a service, a Docker image) and load it
without PyTorch:

```python
from edusync_blueprint_model.predict import OnnxClassifier

clf = OnnxClassifier.from_dir("path/to/quantized")
print(clf.predict(["some text to classify"]))
```

## Git LFS

The model weights are stored with Git LFS (rules in the repo-root `.gitattributes`). Git keeps only a small
pointer file; the real file lives on GitHub's LFS storage.

| Stored in LFS | Stored in normal git |
|---|---|
| `*.onnx`, `*.pt`, `*.pth`, `*.safetensors`, `*.joblib` | JSON files (`labels.json`, `inference_config.json`, `train_config.json`, tokenizer files), reports, READMEs |

**One-time setup on every machine** (before you clone or pull):

```bash
git lfs install
```

If it says `Hook already exists` (for example a Husky hook left by `npm install` in `bigbluebutton-html5`),
use `git lfs install --force`. If you ever see tiny text files starting with
`version https://git-lfs.github.com/spec/v1` instead of models, LFS was not installed when you pulled: run
`git lfs install` then `git lfs pull`.

**Skip the big downloads** (a full checkout downloads about 295 MB of models):

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"; git clone https://github.com/induranga2004/EduTwin-AI.git
git lfs pull --include="edusync/packages/blueprint_model/artifacts/quantized/*"   # only what you need
```

Useful commands: `git lfs ls-files` (what is in LFS), `git lfs status`.

**Quota:** GitHub's free plan gives 1 GB of LFS storage and 1 GB of LFS bandwidth per month, and every full
clone or pull of the models counts against the bandwidth. Commit only models other people need, not every
experiment, and remember that replaced versions stay in the history and keep using storage. When a model
becomes real, record the base model revision, the dataset version and the config next to it
(`train_config.json` already contains the revision and hyperparameters).
