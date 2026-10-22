"""T1 (torch-free parts): config, data, trainer, export contract, quantization logic, reports and
the end-to-end pipeline with a fake torch layer. The real torch layer: test_t1_ml.py."""

import json

import numpy as np
import pytest
from public_fixtures import clear_rows, make_onestop_zip, make_xlsx
from t1_fakes import FakeBackend, FakeImpl, encode_words, make_examples, make_parts

from edusync_c2_model import report
from edusync_c2_model.public_data.clear import load_clear
from edusync_c2_model.public_data.onestop import load_onestop
from edusync_c2_model.public_data.splits import make_group_split, write_split_once
from edusync_c2_model.t1 import data as t1data
from edusync_c2_model.t1 import export as t1export
from edusync_c2_model.t1 import pipeline, quantize, trainer
from edusync_c2_model.t1.config import ConfigError, T1Config

# ------------------------------------------------------------------------------- config


def test_config_defaults_round_trip_and_unknown_keys():
    cfg = T1Config()
    assert (cfg.max_length, cfg.epochs, cfg.batch_size) == (256, 3, 16)
    assert (cfg.lr_encoder, cfg.lr_head) == (3e-5, 1e-3)
    assert T1Config.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ConfigError, match="unknown"):
        T1Config.from_dict({"epochs": 2, "bogus": 1})


@pytest.mark.parametrize(
    "bad",
    [
        {"epochs": 0},
        {"max_length": 0},
        {"lr_head": 0.0},
        {"warmup_ratio": 1.0},
        {"dropout": -0.1},
        {"loss": "l1"},
        {"limit_rows": 0},
    ],
)
def test_config_rejects_bad_values(bad):
    with pytest.raises(ConfigError):
        T1Config(**bad)


# ------------------------------------------------------------------------------- data


def test_pad_batch_pads_dynamically_and_handles_empty():
    batch = t1data.pad_batch([[5, 6, 7], [8]], [1.0, 2.0])
    assert batch.input_ids.tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch.targets.dtype == np.float32
    empty = t1data.pad_batch([])
    assert empty.input_ids.shape == (0, 1)
    assert t1data.pad_batch([[1]], index=[9]).index.tolist() == [9]


def test_bucket_batches_cover_everything_and_group_lengths():
    lengths = [int(x) for x in np.random.default_rng(0).integers(1, 100, 230)]
    for shuffle in (False, True):
        batches = list(t1data.bucket_batches(lengths, 8, seed=3, shuffle=shuffle))
        flat = np.concatenate(batches)
        assert sorted(flat.tolist()) == list(range(230))
        assert all(len(b) <= 8 for b in batches)
    ordered = np.concatenate(list(t1data.bucket_batches(lengths, 8, seed=3, shuffle=False)))
    assert [lengths[i] for i in ordered] == sorted(lengths)
    a = [b.tolist() for b in t1data.bucket_batches(lengths, 8, seed=3, shuffle=True)]
    b = [b.tolist() for b in t1data.bucket_batches(lengths, 8, seed=3, shuffle=True)]
    assert a == b


def test_build_rows_scales_each_source_with_its_own_train_stats():
    parts, easier = make_parts()
    rows, scalers = t1data.build_rows(parts, easier, encode_words)
    assert set(scalers) == {"clear", "onestop"}
    train_clear = [r.target for r in rows["train"] if r.source == "clear"]
    assert np.mean(train_clear) == pytest.approx(0.0, abs=1e-9)
    assert scalers["clear"]["sign"] == -1.0
    assert scalers["onestop"]["sign"] == 1.0
    assert rows["test"][0].ids[0] == 101
    limited, _ = t1data.build_rows(parts, easier, encode_words, limit_rows=3)
    assert len([r for r in limited["train"] if r.source == "clear"]) == 3


def test_predict_rows_restores_the_original_order():
    parts, easier = make_parts()
    rows, _ = t1data.build_rows(parts, easier, encode_words)
    preds = t1data.predict_rows(
        rows["train"], lambda b: b.attention_mask.sum(axis=1).astype(float), 5
    )
    assert preds.tolist() == [float(len(r.ids)) for r in rows["train"]]


# ------------------------------------------------------------------------------- trainer


def test_lr_scale_warms_up_then_decays_to_zero():
    assert trainer.lr_scale(0, 10, 2) == 0.0
    assert trainer.lr_scale(1, 10, 2) == 0.5
    assert trainer.lr_scale(2, 10, 2) == 1.0
    assert trainer.lr_scale(10, 10, 2) == 0.0
    assert trainer.lr_scale(5, 10, 0) == 0.5


def test_mean_spearman_skips_undefined_sources():
    rows = [
        t1data.Row("a", "g1", "t", (1,), 1.0),
        t1data.Row("a", "g2", "t", (1,), 2.0),
        t1data.Row("b", "g3", "t", (1,), 5.0),  # a single row: undefined
    ]
    assert trainer.mean_spearman(rows, np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert trainer.mean_spearman(rows[2:], np.array([1.0])) == 0.0


@pytest.mark.parametrize("improve", [True, False])
def test_fit_selects_best_epoch_and_reads_test_last(improve):
    parts, easier = make_parts()
    rows, _ = t1data.build_rows(parts, easier, encode_words)
    cfg = T1Config(epochs=3, batch_size=4, warmup_ratio=0.2)
    backend = FakeBackend(None, cfg, improve)
    messages = []
    result = trainer.fit(backend, rows, cfg, n_boot=20, log=messages.append)
    assert result["best_epoch"] == 1
    assert [h["epoch"] for h in result["history"]] == [1, 2, 3]
    assert len(messages) == 3
    assert set(result["metrics"]) == {"val", "test"}
    assert set(result["metrics"]["test"]) == {"clear", "onestop"}
    assert backend.scales[0] == 0.0
    assert max(backend.scales) <= 1.0
    steps = -(-len(rows["train"]) // 4) * 3
    assert backend.steps == steps


# ------------------------------------------------------------------------------- export


GOOD_IN = [
    ("input_ids", "tensor(int64)", ["batch", "seq"]),
    ("attention_mask", "tensor(int64)", ["batch", "seq"]),
]
GOOD_OUT = [("score", "tensor(float)", ["batch", 1])]


def test_io_contract_accepts_the_contract_and_names_each_violation():
    t1export.check_io_contract(GOOD_IN, GOOD_OUT)
    bad = t1export.ContractError
    with pytest.raises(bad, match="input names"):
        t1export.check_io_contract(GOOD_IN[::-1], GOOD_OUT)
    with pytest.raises(bad, match="int64"):
        t1export.check_io_contract(
            [("input_ids", "tensor(float)", ["b", "s"]), GOOD_IN[1]], GOOD_OUT
        )
    with pytest.raises(bad, match="int64"):
        t1export.check_io_contract([("input_ids", "tensor(int64)", ["b"]), GOOD_IN[1]], GOOD_OUT)
    with pytest.raises(bad, match="dynamic"):
        t1export.check_io_contract([("input_ids", "tensor(int64)", [1, 8]), GOOD_IN[1]], GOOD_OUT)
    with pytest.raises(bad, match="single tensor"):
        t1export.check_io_contract(GOOD_IN, [("logits", "tensor(float)", ["b", 1])])
    with pytest.raises(bad, match="float32"):
        t1export.check_io_contract(GOOD_IN, [("score", "tensor(double)", ["b", 1])])
    with pytest.raises(bad, match="float32"):
        t1export.check_io_contract(GOOD_IN, [("score", "tensor(float)", ["b", 2])])


def test_check_parity():
    a = np.array([[0.1], [0.2]])
    assert t1export.check_parity(a, a + 5e-4) == pytest.approx(5e-4)
    with pytest.raises(ValueError, match="differs from torch"):
        t1export.check_parity(a, a + 0.1)
    with pytest.raises(ValueError, match="shapes"):
        t1export.check_parity(a, a[:1])


def test_make_predictor_batches_and_parity_check_includes_single_texts():
    seen = []

    def run(feeds):
        seen.append(feeds["input_ids"].shape)
        return feeds["attention_mask"].sum(axis=1, keepdims=True).astype(np.float32)

    predict = t1export.make_predictor(encode_words, run, batch_size=2)
    out = predict(["a b", "a b c d", "a"])
    assert out.tolist() == [4.0, 6.0, 3.0]
    assert [s[0] for s in seen] == [2, 1]

    def torch_like(texts):
        return np.array([len(i) for i in encode_words(texts)], dtype=float)

    assert t1export.parity_check(torch_like, predict) == 0.0
    with pytest.raises(ValueError, match="differs"):
        t1export.parity_check(lambda t: torch_like(t) + 1, predict)


def test_write_sidecars_and_inference_config(tmp_path):
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}", encoding="utf-8")
    scalers = {"clear": {"mean": 1.5, "std": 2.0, "sign": -1.0, "n_train": 9, "version": 1}}
    t1export.write_sidecars(tmp_path / "out", tok, T1Config(max_length=192), scalers)
    info = json.loads((tmp_path / "out" / "inference_config.json").read_text(encoding="utf-8"))
    assert (tmp_path / "out" / "tokenizer.json").read_text(encoding="utf-8") == "{}"
    assert info["max_length"] == 192
    assert info["target"] == "difficulty_z"
    assert info["train_mean_std"] == {"clear": {"mean": 1.5, "std": 2.0}}
    assert info["base_revision"] == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert "CC BY-NC-SA" in info["licence"]
    assert "CLEAR" in info["trained_on"]


# ------------------------------------------------------------------------------- quantize


def test_remove_new_files_keeps_listed_and_pre_existing(tmp_path):
    old = tmp_path / "old.onnx"
    old.write_bytes(b"1")
    before = quantize.snapshot_dir(tmp_path)
    keep = tmp_path / "keep.onnx"
    keep.write_bytes(b"1")
    (tmp_path / "leak.data").write_bytes(b"1")
    (tmp_path / "sub").mkdir()
    assert quantize.remove_new_files(tmp_path, before, keep=[keep]) == ["leak.data"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.onnx", "old.onnx", "sub"]


def test_latency_per_text_divides_by_batch_size():
    ticks = iter([0.0, 2.0])
    calls = []
    value = quantize.latency_ms_per_text(
        lambda b: calls.append(len(b)), ["a", "b"], 4, 5, timer=lambda: next(ticks)
    )
    assert value == pytest.approx(2000.0 / (5 * 4))
    assert calls == [4] * 6  # one warm-up plus five timed runs


@pytest.mark.parametrize(
    ("drop", "ship"),
    [(0.0, "int8"), (0.019, "int8"), (0.02, "fp32"), (0.1, "fp32"), (None, "fp32")],
)
def test_decide_ship_rule(drop, ship):
    decision = quantize.decide_ship(drop, 0.02)
    assert decision["ship"] == ship
    assert decision["drop"] == drop
    assert decision["reason"]


def test_compare_models_reports_quality_size_latency_and_decision(tmp_path):
    fp32, int8 = tmp_path / "a.onnx", tmp_path / "b.onnx"
    fp32.write_bytes(b"x" * 2048)
    int8.write_bytes(b"x" * 512)
    texts = [f"t{i}" for i in range(12)]
    y = np.arange(12, dtype=float)
    groups = [f"g{i % 4}" for i in range(12)]

    def good(t):
        return np.array([float(x[1:]) for x in t])

    def noisy(t):
        return good(t) * -1

    sets = {"s": (texts, y, groups)}
    kwargs = {"fp32_path": fp32, "int8_path": int8, "threshold": 0.02, "runs": 2}
    same = quantize.compare_models(good, good, sets, **kwargs)
    assert same["decision"]["ship"] == "int8"
    assert same["quality"]["s"]["max_abs_diff"] == 0.0
    assert set(same["latency_ms_per_text"]) == {1, 16}
    assert same["size_mb"]["fp32"] == pytest.approx(2048 / 1048576)
    worse = quantize.compare_models(good, noisy, sets, **kwargs)
    assert worse["decision"]["ship"] == "fp32"
    constant = quantize.compare_models(good, lambda t: np.zeros(len(t)), sets, **kwargs)
    assert constant["decision"]["ship"] == "fp32"
    assert constant["decision"]["drop"] is None


# ------------------------------------------------------------------------------- pipeline


def test_run_end_to_end_with_fake_torch_layer(tmp_path):
    parts, easier = make_parts()
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    impl = FakeImpl()
    cfg = T1Config(epochs=2, batch_size=4, threads=3, limit_rows=None)
    messages = []
    summary = pipeline.run(
        cfg, parts, easier, base_dir=base, out_dir=out, impl=impl, n_boot=20, log=messages.append
    )
    assert ("threads", 3) in impl.calls
    assert ("seed", 42) in impl.calls
    assert summary["removed_temp_files"] == ["model_prep.onnx"]
    assert summary["parity_max_abs_diff"] < 1e-6
    assert summary["quantization"]["decision"]["ship"] == "int8"
    assert set(summary["quantization"]["quality"]) == {"clear test", "onestop test"}
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "inference_config.json",
        "model_card.md",
        "model_fp32.onnx",
        "model_int8.onnx",
        "report.md",
        "tokenizer.json",
        "train_summary.json",
    ]
    saved = json.loads((out / "train_summary.json").read_text(encoding="utf-8"))
    assert saved["versions"] == {"fake": "1"}
    assert saved["config"]["epochs"] == 2
    assert "Decision" in (out / "report.md").read_text(encoding="utf-8")
    assert "Shipped file: int8" in (out / "model_card.md").read_text(encoding="utf-8")
    assert messages


def test_run_smoke_limit_rows(tmp_path):
    parts, easier = make_parts()
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text("{}", encoding="utf-8")
    summary = pipeline.run(
        T1Config(epochs=1, batch_size=2, limit_rows=4),
        parts,
        easier,
        base_dir=base,
        out_dir=tmp_path / "o",
        impl=FakeImpl(),
        n_boot=5,
        log=lambda _m: None,
    )
    assert summary["n_train"] == 8


def test_load_sources_reads_sealed_splits_and_reports_missing(tmp_path):
    with pytest.raises(pipeline.DataMissingError):
        pipeline.load_sources(tmp_path)
    make_xlsx(tmp_path / "CLEAR_corpus_final.xlsx", clear_rows(30))
    make_onestop_zip(tmp_path / "onestop.zip", 12)
    (tmp_path / "splits").mkdir()
    for key, loader in (("clear", load_clear), ("onestop", load_onestop)):
        examples = loader(
            tmp_path / {"clear": "CLEAR_corpus_final.xlsx", "onestop": "onestop.zip"}[key]
        )
        split = make_group_split({e.group_id for e in examples}, seed=1, source=key)
        write_split_once(tmp_path / "splits" / f"{key}_split.json", split)
    parts, easier = pipeline.load_sources(tmp_path)
    assert set(parts) == {"clear", "onestop"}
    assert easier == {"clear": True, "onestop": False}
    assert sum(len(v) for v in parts["clear"].values()) == 30


# ------------------------------------------------------------------------------- reports


def test_reports_render_tables_and_handle_missing_values():
    ev = {
        "n": 5,
        "spearman": 0.5,
        "rmse": None,
        "ci": {"spearman": {"low": 0.1, "high": 0.9}, "rmse": None},
    }
    table = report.render_quality_table({"test": {"clear": ev}, "val": {"onestop": ev}})
    assert "| test | clear | 5 | 0.500 | 0.100 to 0.900 | n/a | n/a |" in table
    comparison = {
        "quality": {
            "clear test": {"n": 5, "fp32": ev, "int8": ev, "max_abs_diff": 0.01},
        },
        "size_mb": {"fp32": 90.0, "int8": 0.0},
        "latency_ms_per_text": {1: {"fp32": 5.0, "int8": 3.0}},
        "decision": {"ship": "fp32", "reason": "because"},
        "threshold": 0.02,
    }
    md = report.render_quantization_table(comparison)
    assert "0.00x smaller" in md
    assert "ship **fp32**" in md
    assert "batch 1" in md
    comparison["size_mb"]["int8"] = 30.0
    assert "3.00x smaller" in report.render_quantization_table(comparison)
    summary = {
        "best_epoch": 2,
        "n_train": 10,
        "history": [{"epoch": 1, "train_loss": 0.5, "val_spearman": None, "seconds": 3.2}],
        "metrics": {"test": {"clear": ev}, "val": {"clear": ev}},
        "quantization": comparison,
    }
    assert "Best epoch: 2" in report.render_t1_report(summary)
    card = report.render_model_card(
        summary,
        {"output": "score", "trained_on": "x", "max_length": 8, "licence": "NC"},
    )
    assert "clear test Spearman 0.500" in card
    assert "Licence: NC" in card


def test_make_examples_helper_is_deterministic():
    assert make_examples("clear", 3) == make_examples("clear", 3)
