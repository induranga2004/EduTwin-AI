# Artifacts

Everything the blueprint produces is written here. The folder is **git-ignored** (only `.gitkeep` and this
README are tracked), so models never end up in the repository.

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

## Sharing models

Do not commit these files. Share them through your team's storage (a release asset, a drive folder, a model
registry) and record the base model revision and the dataset version next to them; `train_config.json`
already contains the revision and hyperparameters.
