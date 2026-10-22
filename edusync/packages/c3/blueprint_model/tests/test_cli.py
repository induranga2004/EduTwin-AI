import json
from pathlib import Path

import numpy as np
import pytest

from edusync_blueprint_model import cli
from edusync_blueprint_model.deps import MissingExtraError
from edusync_blueprint_model.predict import OnnxClassifier, Prediction
from edusync_blueprint_model.train import TrainResult


class FakeClf(OnnxClassifier):
    def __init__(self):
        self.labels = ["a", "b", "c"]
        self.max_length = 8

    def logits(self, texts):
        out = np.zeros((len(texts), 3))
        for i, t in enumerate(texts):
            out[i, 0 if t.endswith("a") else 1] = 1.0
        return out


def run(config_file: Path, *argv: str) -> int:
    return cli.main(["--config", str(config_file), *argv])


def raiser(exc: Exception):
    def _raise(*args, **kwargs):
        raise exc

    return _raise


def test_info_without_extras(config_file, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "is_available", lambda name: False)
    assert run(config_file, "info") == 0
    out = capsys.readouterr().out
    assert "org/tiny-model" in out
    assert "missing" in out
    assert "--extra finetune" in out
    assert "not run" in out


def test_info_all_installed_and_done(config_file, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "is_available", lambda name: True)
    cfg = cli.load_config(config_file)
    marker = cfg.artifacts.finetuned_dir / "head.pt"
    marker.parent.mkdir(parents=True)
    marker.write_text("x")
    assert run(config_file, "info") == 0
    out = capsys.readouterr().out
    assert "--extra" not in out
    assert "done" in out


def test_python_dash_m_entry() -> None:
    import edusync_blueprint_model.__main__ as m

    assert m.main is cli.main


def test_bad_args_exit() -> None:
    with pytest.raises(SystemExit):
        cli.main(["bogus"])


def test_parser_defaults() -> None:
    p = cli.build_parser()
    assert p.parse_args(["predict", "hi"]).model == "int8"
    assert p.parse_args(["evaluate"]).model == "fp32"
    assert p.parse_args(["evaluate", "--model", "int8"]).model == "int8"


def test_missing_extra_error(config_file, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "download_base", raiser(MissingExtraError("install me")))
    assert run(config_file, "download") == 3
    assert "install me" in capsys.readouterr().err


def test_missing_file_gives_hint(config_file, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "run_train", raiser(FileNotFoundError("no weights")))
    assert run(config_file, "train") == 4
    captured = capsys.readouterr()
    assert "no weights" in captured.err
    assert "download" in captured.out


def test_missing_file_without_hint(config_file, monkeypatch) -> None:
    monkeypatch.setattr(cli, "download_base", raiser(FileNotFoundError("x")))
    assert run(config_file, "download") == 4


def test_bad_config_and_value_errors(tmp_path, config_file, monkeypatch, capsys) -> None:
    assert cli.main(["--config", str(tmp_path / "none.toml"), "info"]) == 2
    monkeypatch.setattr(cli, "export_onnx", raiser(ValueError("bad")))
    assert run(config_file, "export") == 2
    assert "bad" in capsys.readouterr().err


def test_download_and_export_commands(config_file, monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setattr(cli, "download_base", lambda c: tmp_path / "base")
    monkeypatch.setattr(cli, "export_onnx", lambda c: tmp_path / "m.onnx")
    assert run(config_file, "download") == 0
    assert run(config_file, "export") == 0
    out = capsys.readouterr().out
    assert "base" in out
    assert "m.onnx" in out


def test_train_command(config_file, monkeypatch, capsys, tmp_path) -> None:
    result = TrainResult(
        best_epoch=1,
        best_val_macro_f1=0.5,
        test={"accuracy": 0.6, "macro_f1": 0.5},
        history=[{"epoch": 1, "train_loss": 1.0, "val_accuracy": 0.5, "val_macro_f1": 0.5}],
        finetuned_dir=tmp_path,
    )
    monkeypatch.setattr(cli, "run_train", lambda c: result)
    assert run(config_file, "train") == 0
    assert "best epoch 1" in capsys.readouterr().out


def test_evaluate_command(config_file, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_load_classifier", lambda c, kind: FakeClf())
    assert run(config_file, "evaluate", "--model", "int8") == 0
    out = capsys.readouterr().out
    assert "accuracy" in out
    assert "int8 on 6 test examples" in out
    report = config_file.parent.parent / "out" / "reports" / "evaluation_int8.json"
    assert json.loads(report.read_text())["num_examples"] == 6


def test_quantize_command(config_file, monkeypatch, capsys) -> None:
    cfg = cli.load_config(config_file)
    fp32 = cfg.artifacts.onnx_dir / "model_fp32.onnx"
    int8 = cfg.artifacts.quantized_dir / "model_int8.onnx"
    for f, size in ((fp32, 400), (int8, 100)):
        f.parent.mkdir(parents=True)
        f.write_bytes(b"x" * size)
    monkeypatch.setattr(cli, "quantize_model", lambda c: int8)
    monkeypatch.setattr(cli, "_load_classifier", lambda c, kind: FakeClf())
    assert run(config_file, "quantize") == 0
    out = capsys.readouterr().out
    assert "size (MB)" in out
    assert "agreement" in out
    assert (cfg.artifacts.reports_dir / "quantization_report.json").is_file()


def test_predict_command(config_file, monkeypatch, capsys, tmp_path) -> None:
    class Clf:
        def predict(self, texts, positive_label=None, threshold=0.5):
            return [Prediction(label="a", score=0.9, probabilities={"a": 0.9}) for _ in texts]

    seen = []

    def loader(config, kind):
        seen.append(kind)
        return Clf()

    monkeypatch.setattr(cli, "_load_classifier", loader)
    texts = tmp_path / "t.txt"
    texts.write_text("line one\n\nline two\n")
    assert run(config_file, "predict", "--model", "fp32", "direct", "--file", str(texts)) == 0
    assert capsys.readouterr().out.count("a\t0.900") == 3
    assert seen == ["fp32"]
    assert run(config_file, "predict") == 2


def test_model_dir_and_loader(config_file, monkeypatch) -> None:
    cfg = cli.load_config(config_file)
    assert cli._model_dir(cfg, "fp32") == cfg.artifacts.onnx_dir
    assert cli._model_dir(cfg, "int8") == cfg.artifacts.quantized_dir
    got = {}

    def fake_from_dir(cls, directory):
        got["d"] = directory
        return "clf"

    monkeypatch.setattr(OnnxClassifier, "from_dir", classmethod(fake_from_dir))
    assert cli._load_classifier(cfg, "int8") == "clf"
    assert got["d"] == cfg.artifacts.quantized_dir
