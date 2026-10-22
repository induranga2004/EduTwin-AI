"""T1 with the real torch / onnxruntime layer. Deselected in CI; run inside the training container:
``bash scripts/docker_train_t1.sh --pytest`` (tiny model inputs: 64 rows, one epoch)."""

from pathlib import Path

import numpy as np
import pytest
from t1_fakes import make_parts

pytestmark = pytest.mark.ml

BASE = (
    Path(__file__).resolve().parents[2]
    / "blueprint_model"
    / "artifacts"
    / "base"
    / "sentence-transformers--all-MiniLM-L6-v2"
)


@pytest.fixture(scope="module")
def torch_impl():
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from edusync_c2_model.t1 import torch_impl as module

    return module


@pytest.fixture(scope="module")
def tiny_run(torch_impl, tmp_path_factory):
    from edusync_c2_model.t1 import pipeline
    from edusync_c2_model.t1.config import T1Config

    parts, easier = make_parts()
    out = tmp_path_factory.mktemp("t1")
    config = T1Config(epochs=1, max_length=32, batch_size=8, threads=2, limit_rows=64)
    summary = pipeline.run(
        config, parts, easier, base_dir=BASE, out_dir=out, impl=torch_impl, n_boot=20, log=print
    )
    return out, summary


def test_encoder_truncates_and_adds_special_tokens(torch_impl):
    encode = torch_impl.load_encoder(BASE / "tokenizer.json", 8)
    ids = encode(["hello world", "a much longer sentence that must be cut to eight tokens"])
    assert ids[0][0] == 101
    assert ids[0][-1] == 102
    assert len(ids[1]) == 8


def test_model_forward_shapes_and_backend_snapshot(torch_impl):
    from edusync_c2_model.t1.config import T1Config
    from edusync_c2_model.t1.data import pad_batch

    model = torch_impl.build_model(BASE, 0.1)
    backend = torch_impl.TorchBackend(model, T1Config(epochs=1, freeze_encoder=True))
    batch = pad_batch([[101, 2000, 102], [101, 102]], [0.5, -0.5])
    assert backend.predict_batch(batch).shape == (2,)
    before = backend.snapshot()
    assert np.isfinite(backend.train_step(batch, 1.0))
    backend.restore(before)
    assert all(not p.requires_grad for n, p in model.named_parameters() if n.startswith("encoder."))


def test_smoke_pipeline_writes_all_artifacts(tiny_run):
    out, summary = tiny_run
    for name in (
        "model_fp32.onnx",
        "model_int8.onnx",
        "tokenizer.json",
        "inference_config.json",
        "model_card.md",
    ):
        assert (out / name).is_file()
    assert summary["parity_max_abs_diff"] <= 1e-3
    assert summary["removed_temp_files"] is not None
    assert not [p for p in out.iterdir() if p.name == "model_prep.onnx" or p.suffix == ".data"]
    assert (out / "model_int8.onnx").stat().st_size < (out / "model_fp32.onnx").stat().st_size


def test_onnx_handles_batch_of_one_and_other_lengths(torch_impl, tiny_run):
    from edusync_c2_model.t1.export import make_predictor

    out, _ = tiny_run
    run, _, _ = torch_impl.open_session(out / "model_fp32.onnx", 2)
    encode = torch_impl.load_encoder(out / "tokenizer.json", 32)
    predict = make_predictor(encode, run)
    one = predict(["Short."])
    both = predict(["Short.", "A rather longer sentence with quite a few more words in it today."])
    assert one.shape == (1,)
    assert abs(one[0] - both[0]) < 1e-3
