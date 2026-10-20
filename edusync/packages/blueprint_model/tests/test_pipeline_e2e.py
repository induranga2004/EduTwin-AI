"""Full pipeline on the sample CSV with the real model (needs the finetune extra + network)."""

import shutil
import sys
from pathlib import Path

import pytest

from edusync_blueprint_model.config import DEFAULT_CONFIG_PATH, load_config
from edusync_blueprint_model.download import download_base
from edusync_blueprint_model.onnx_export import export_onnx
from edusync_blueprint_model.predict import OnnxClassifier
from edusync_blueprint_model.quantize import quantize_model
from edusync_blueprint_model.train import train

pytestmark = pytest.mark.ml


def test_download_train_export_quantize_predict(tmp_path: Path) -> None:
    sample = DEFAULT_CONFIG_PATH.parent.parent / "data" / "sample.csv"
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace("epochs = 4", "epochs = 1").replace(
        "benchmark_runs = 50", "benchmark_runs = 2"
    )
    text = text.replace('"../data/sample.csv"', f'"{sample.as_posix()}"')
    text = text.replace('"../artifacts"', f'"{(tmp_path / "artifacts").as_posix()}"')
    cfg_file = tmp_path / "bp.toml"
    cfg_file.write_text(text, encoding="utf-8")
    cfg = load_config(cfg_file)

    # Reuse an already-downloaded base model if there is one, to save time.
    cached = load_config().artifacts.base_model_dir(cfg.model.name)
    if (cached / "model.safetensors").is_file():
        shutil.copytree(cached, cfg.artifacts.base_model_dir(cfg.model.name))
    download_base(cfg)

    result = train(cfg)
    assert (result.finetuned_dir / "head.pt").is_file()
    fp32 = export_onnx(cfg)
    int8 = quantize_model(cfg)
    assert int8.stat().st_size < fp32.stat().st_size

    for folder in (cfg.artifacts.onnx_dir, cfg.artifacts.quantized_dir):
        preds = OnnxClassifier.from_dir(folder).predict(["hello there", "another text"])
        assert len(preds) == 2
        assert sum(preds[0].probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    assert "OnnxClassifier" in dir(sys.modules["edusync_blueprint_model.predict"])
