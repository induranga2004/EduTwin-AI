"""T2 numpy text model: chunking, fit, predict, persistence and the training script."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from public_fixtures import clear_rows, make_onestop_zip, make_xlsx

from edusync_c2_model import t2
from edusync_c2_model.regression_metrics import spearman

PKG = Path(__file__).resolve().parents[1]
MODEL_DIR = PKG.parent / "blueprint_model/artifacts/base/sentence-transformers--all-MiniLM-L6-v2"


def fake_embed(texts):
    """Deterministic hash-based unit vectors (order independent)."""
    out = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")
        out.append(np.random.default_rng(seed).normal(size=8))
    return _unit(np.array(out))


def _unit(v):
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def corpus(n=40):
    texts = []
    for i in range(n):
        sentence = "The cat sat on the mat. " * (1 + i % 5)
        texts.append(sentence + f"Passage {i} is here.")
    words = np.array([len(t.split()) for t in texts], dtype=float)
    return texts, (words - words.mean()) / words.std()


# ------------------------------------------------------------------------------ chunking


def test_chunk_text_windows_and_fallbacks():
    text = "One two three. Four five six. Seven eight nine ten."
    assert t2.chunk_text(text, 6) == ["One two three Four five six", "Seven eight nine ten"]
    assert t2.chunk_text(text, 100) == ["One two three Four five six Seven eight nine ten"]
    long_sentence = "a b c d e f g"
    assert t2.chunk_text("x y. " + long_sentence + ".", 3) == ["x y", "a b c", "d e f", "g"]
    assert t2.chunk_text("  ", 5) == ["  "]
    with pytest.raises(t2.T2Error, match="max_words"):
        t2.chunk_text("a b", 0)


def test_embed_texts_variants():
    texts = ["First part here. Second part there.", "Short one."]
    first = t2.embed_texts(fake_embed, texts, None)
    assert first.shape == (2, 8)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0)
    chunked = t2.embed_texts(fake_embed, texts, 3)
    expected = fake_embed(["First part here", "Second part there"]).mean(axis=0)
    assert np.allclose(chunked[0], expected / np.linalg.norm(expected))
    assert np.allclose(chunked[1], fake_embed(["Short one"])[0])
    assert t2.embed_texts(fake_embed, [], 3).shape == (0, 0)
    assert np.allclose(t2.embed_texts(lambda t: np.zeros((len(t), 4)), ["a"], None), 0.0)


# ---------------------------------------------------------------------------- fit / predict


@pytest.mark.parametrize("chunk", [None, 12])
def test_fit_predict_learns_and_is_deterministic(chunk):
    texts, y = corpus()
    model = t2.T2Model(fake_embed, "sha", chunk).fit(texts, y, seed=1)
    z, unc = model.predict(texts)
    assert spearman(y, z) > 0.8
    assert z.shape == unc.shape == (40,)
    assert np.all(unc >= model.cv_rmse - 1e-12)
    again = t2.T2Model(fake_embed, "sha", chunk).fit(texts, y, seed=1)
    assert np.array_equal(again.predict(texts)[0], z)
    assert model.n_train == 40
    assert model.seed == 1
    assert model.alpha in t2.ALPHA_GRID


def test_fit_is_shuffle_invariant():
    texts, y = corpus()
    groups = [f"g{i % 8}" for i in range(len(texts))]
    base = t2.T2Model(fake_embed, "sha", None).fit(texts, y, 3, groups)
    order = np.random.default_rng(0).permutation(len(texts))
    shuffled = t2.T2Model(fake_embed, "sha", None).fit(
        [texts[i] for i in order], y[order], 3, [groups[i] for i in order]
    )
    assert shuffled.alpha == base.alpha
    assert np.allclose(shuffled.predict(texts)[0], base.predict(texts)[0], atol=1e-8)


def test_fit_edge_cases():
    texts, y = corpus(12)
    single_group = t2.T2Model(fake_embed, "sha", None).fit(texts, y, 0, ["g"] * 12)
    assert single_group.cv_rmse >= 0.0
    two_groups = t2.T2Model(fake_embed, "sha", None).fit(texts, y, 0, ["a", "b"] * 6, alphas=(5.0,))
    assert two_groups.alpha == 5.0
    same = t2.T2Model(fake_embed, "sha", None).fit(["Same text."] * 4, [0.0, 1.0, 2.0, 3.0], 0)
    assert same.median_dist == 1.0
    assert same.predict([])[0].size == 0
    with pytest.raises(t2.T2Error, match="same length"):
        t2.T2Model(fake_embed, "s", None).fit(texts, y[:-1], 0)
    with pytest.raises(t2.T2Error, match="same length"):
        t2.T2Model(fake_embed, "s", None).fit(texts, y, 0, ["g"])
    with pytest.raises(t2.T2Error, match="at least 2"):
        t2.T2Model(fake_embed, "s", None).fit(texts[:1], y[:1], 0)
    with pytest.raises(t2.T2Error, match="at least 2"):
        t2.T2Model(fake_embed, "s", None).fit(texts, [np.nan] * 12, 0)


# ----------------------------------------------------------------------------- persistence


def test_npz_round_trip_and_files(tmp_path):
    texts, y = corpus()
    model = t2.T2Model(fake_embed, "abc123", 12).fit(texts, y, 5)
    data = model.to_npz_dict()
    assert data["config_json"].dtype.kind == "U"
    config = json.loads(str(data["config_json"]))
    assert config["encoder_sha"] == "abc123"
    assert config["chunk_words"] == 12
    assert config["simple_feature_names"][0] == "fk_grade"
    assert "non-commercial" in config["licence_note"]
    rebuilt = t2.T2Model.from_npz_dict(data, fake_embed)
    assert np.array_equal(rebuilt.predict(texts)[0], model.predict(texts)[0])
    npz, cfg = model.save(tmp_path / "out")
    assert json.loads(cfg.read_text(encoding="utf-8")) == model.config()
    loaded = t2.T2Model.load(tmp_path / "out", fake_embed)
    assert np.array_equal(loaded.predict(texts)[1], model.predict(texts)[1])
    assert npz.name == t2.NPZ_NAME


def test_load_errors(tmp_path):
    texts, y = corpus(12)
    model = t2.T2Model(fake_embed, "s", None).fit(texts, y, 0)
    data = model.to_npz_dict()
    with pytest.raises(t2.T2Error, match="invalid"):
        t2.T2Model.from_npz_dict({k: v for k, v in data.items() if k != "coef"})
    with pytest.raises(t2.T2Error, match="invalid"):
        t2.T2Model.from_npz_dict({**data, "config_json": np.array("{nope")})
    bad_kind = json.loads(str(data["config_json"])) | {"model_kind": "other"}
    with pytest.raises(t2.T2Error, match="unsupported"):
        t2.T2Model.from_npz_dict({**data, "config_json": np.array(json.dumps(bad_kind))})
    with pytest.raises(t2.T2Error, match="do not match"):
        t2.T2Model.from_npz_dict({**data, "coef": data["coef"][:-1]})
    with pytest.raises(t2.T2Error, match="cannot read"):
        t2.T2Model.load(tmp_path / "missing")
    no_embedder = t2.T2Model.from_npz_dict(data)
    with pytest.raises(t2.T2Error, match="without an embedder"):
        no_embedder.predict(["text"])


@pytest.mark.ml
def test_real_encoder_smoke():
    if not (MODEL_DIR / "model.safetensors").is_file():
        pytest.skip("MiniLM weights not present")
    from edusync_c2_model.models.embeddings import encoder_embedder

    embed, sha = encoder_embedder(MODEL_DIR)
    texts, y = corpus(20)
    model = t2.T2Model(embed, sha, 50).fit(texts, y, 0)
    z, _ = model.predict(texts)
    assert z.shape == (20,)
    assert np.all(np.isfinite(z))


# ------------------------------------------------------------------------------------ script


def load_script():
    spec = importlib.util.spec_from_file_location("train_t2", PKG / "scripts" / "train_t2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data(tmp_path, *, clear=True, onestop=True):
    data_dir = tmp_path / "pub"
    data_dir.mkdir()
    if clear:
        make_xlsx(data_dir / "CLEAR_corpus_final.xlsx", clear_rows(60))
    if onestop:
        make_onestop_zip(data_dir / "onestop.zip", 30)
    return data_dir


def test_script_run_writes_artifact_and_report(tmp_path):
    script = load_script()
    data_dir = make_data(tmp_path)
    runs = tmp_path / "runs"
    logs = []
    summary = script.run(
        data_dir,
        runs,
        embedder=(fake_embed, "fakesha"),
        cache_dir=tmp_path / "cache",
        n_boot=20,
        chunk_words=10,
        log=logs.append,
    )
    assert len(summary["terciles"]) == 6
    assert len(logs) == 6
    assert summary["best"] in summary["alpha"]
    report = (runs / "t2_report.md").read_text(encoding="utf-8")
    assert "Low/Medium/High mapping check" in report
    assert "baseline_ridge_all" in report
    card = (runs / "t2" / "model_card.md").read_text(encoding="utf-8")
    assert "non-commercial" in card
    assert "DRAFT" in card
    model = t2.T2Model.load(runs / "t2", fake_embed)
    assert model.encoder_sha == "fakesha"
    assert (tmp_path / "cache" / "embeddings-fakesha.npz").is_file()


def test_script_single_corpus_and_missing_data(tmp_path, capsys):
    script = load_script()
    only_clear = make_data(tmp_path, onestop=False)
    summary = script.run(
        only_clear, tmp_path / "r", embedder=(fake_embed, "s"), cache_dir=None, n_boot=10
    )
    assert set(summary["sizes"]) == {"clear"}
    assert len(summary["terciles"]) == 2
    only_onestop = tmp_path / "os"
    only_onestop.mkdir()
    make_onestop_zip(only_onestop / "onestop.zip", 30)
    summary = script.run(
        only_onestop, tmp_path / "r2", embedder=(fake_embed, "s"), cache_dir=None, n_boot=10
    )
    assert summary["terciles"] == {}
    empty = tmp_path / "empty"
    empty.mkdir()
    assert script.run(empty, tmp_path / "r3", embedder=(fake_embed, "s"), cache_dir=None) is None
    assert "skipped" in capsys.readouterr().out


def test_script_helpers_with_missing_values():
    script = load_script()
    assert script._num(None) == "n/a"
    assert script._ci(None) == "n/a"
    rows = [
        {"name": "a", "part": "val", "spearman": None},
        {"name": "b", "part": "val", "spearman": 0.2},
        {"name": "b", "part": "test", "spearman": 0.9},
    ]
    assert script._choose_best(rows, {"a": 1, "b": 2}) == "b"
    assert script._choose_best([], {"b": 1, "a": 2}) == "a"


def test_script_main(tmp_path, monkeypatch, capsys):
    script = load_script()
    data_dir = make_data(tmp_path, onestop=False)
    monkeypatch.setattr(script, "encoder_embedder", lambda _dir: (fake_embed, "ms"))
    args = [
        "--data-dir",
        str(data_dir),
        "--runs-dir",
        str(tmp_path / "runs"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--n-boot",
        "10",
    ]
    assert script.main(args) == 0
    assert "artifact" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert script.main(["--data-dir", str(empty), "--runs-dir", str(tmp_path / "r")]) == 0
    assert "nothing trained" in capsys.readouterr().out

    def boom(_dir):
        raise RuntimeError("no weights")

    monkeypatch.setattr(script, "encoder_embedder", boom)
    assert script.main(args) == 1
    assert "no weights" in capsys.readouterr().err
