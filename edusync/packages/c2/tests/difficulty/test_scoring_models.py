import sys
import types

import numpy as np
import pytest
from scoring_helpers import (
    fake_embed,
    fake_session_factory,
    fake_tokenizer_factory,
    write_artifact_dir,
    write_t1,
    write_t2,
)

from edusync_c2.difficulty.features.vocab import get_default_analyzer
from edusync_c2.difficulty.scoring import artifact as art
from edusync_c2.difficulty.scoring import text_models as tm
from edusync_c2.difficulty.scoring.reference import ScoringError

TEXT = (
    "A hash function maps data to a fixed size digest. It is easy to compute. Hashing hides input."
)


def test_helpers_syllables_sentences_chunks():
    assert tm._syllables("") == 0
    assert tm._syllables("make") == 1
    assert tm._syllables("table") == 2
    assert tm._syllables("jumped") == 1
    assert tm._syllables("wanted") == 2
    chunks = tm.chunk_text(TEXT, 8)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 8 for c in chunks)
    assert tm.chunk_text("x" * 5, 3) == ["xxxxx"]
    assert tm.chunk_text("", 3) == [""]
    assert tm.chunk_text("a b c d e f g. h i.", 4)  # long sentence split in windows
    unit = tm.embed_texts(fake_embed, ["a b", "c d e"], None)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0)
    pooled = tm.embed_texts(fake_embed, [TEXT, "short"], 6)
    assert pooled.shape == (2, 4)


def test_simple_features():
    an = get_default_analyzer()
    assert (tm.simple_features("!!!", an) == 0).all()
    v = tm.simple_features(TEXT, an)
    assert v.shape == (5,)
    assert v[2] > 3
    assert tm.simple_features("a b", an)[3:].tolist() == [0.0, 1.0]  # no analysable words


def test_t2_scorer(tmp_path, monkeypatch):
    write_t2(tmp_path)
    scorer = tm.T2Scorer.load(tmp_path, fake_embed, [-1.0, 0.0, 1.0])
    assert scorer.z([]).shape == (0,)
    z = scorer.z([TEXT, "other words here"])
    assert z.shape == (2,)
    assert np.isfinite(z).all()
    assert all(0 <= p <= 1 for p in scorer.percentiles([TEXT]))
    write_t2(tmp_path, chunk_words=None)
    assert tm.T2Scorer.load(tmp_path, fake_embed, [0.0]).chunk_words is None
    write_t2(tmp_path, kind="other")
    with pytest.raises(ScoringError):
        tm.T2Scorer.load(tmp_path, fake_embed, [0.0])
    write_t2(tmp_path, dims_ok=False)
    with pytest.raises(ScoringError):
        tm.T2Scorer.load(tmp_path, fake_embed, [0.0])
    with pytest.raises(ScoringError):
        tm.T2Scorer.load(tmp_path / "nowhere", fake_embed, [0.0])
    write_t2(tmp_path)
    monkeypatch.setattr(tm, "get_default_analyzer", lambda: None)
    with pytest.raises(ScoringError):
        tm.T2Scorer.load(tmp_path, fake_embed, [0.0])


def test_t1_scorer(tmp_path):
    write_t1(tmp_path)
    scorer = tm.T1Scorer.load(
        tmp_path,
        [0.0, 0.5, 1.0],
        session_factory=fake_session_factory,
        tokenizer_factory=fake_tokenizer_factory,
    )
    texts = [f"text number {i} " * (i % 4 + 1) for i in range(20)]  # more than one batch
    z = scorer.z(texts)
    assert z.shape == (20,)
    assert (z > 0).all()
    assert len(scorer.percentiles(texts[:3])) == 3
    assert scorer.model_file == "t1_int8.onnx"


def test_t1_load_errors(tmp_path):
    with pytest.raises(ScoringError):
        tm.T1Scorer.load(tmp_path, [0.0])
    write_t1(tmp_path, fp32_only=True)
    ok = tm.T1Scorer.load(
        tmp_path,
        [0.0],
        session_factory=fake_session_factory,
        tokenizer_factory=fake_tokenizer_factory,
    )
    assert ok.model_file == "t1_fp32.onnx"

    def boom(_path):
        raise RuntimeError("no")

    with pytest.raises(ScoringError):
        tm.T1Scorer.load(
            tmp_path, [0.0], session_factory=boom, tokenizer_factory=fake_tokenizer_factory
        )
    (tmp_path / "inference_config.json").unlink()
    with pytest.raises(ScoringError):
        tm.T1Scorer.load(tmp_path, [0.0], session_factory=fake_session_factory)


def test_default_factories_with_fake_modules(monkeypatch, tmp_path):
    seen = {}

    class Options:
        intra_op_num_threads = 0

    class Session:
        def __init__(self, path, options, providers):
            seen["args"] = (path, options.intra_op_num_threads, providers)

    ort = types.SimpleNamespace(SessionOptions=Options, InferenceSession=Session)

    class Tok:
        def no_padding(self):
            seen["pad"] = False

        def enable_truncation(self, max_length):
            seen["trunc"] = max_length

        def encode_batch(self, texts):
            return [types.SimpleNamespace(ids=[1, len(t)]) for t in texts]

    class Tokenizer:
        @staticmethod
        def from_file(path):
            return Tok()

    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=Tokenizer))
    tm.default_session_factory(tmp_path / "m.onnx")
    assert seen["args"][1:] == (4, ["CPUExecutionProvider"])
    encode = tm.default_tokenizer_factory(tmp_path / "t.json", 7)
    assert seen["trunc"] == 7
    assert encode(["ab"]) == [[1, 2]]


# -------------------------------------------------------------------------------- artifact


class Encoder:
    encode = staticmethod(fake_embed)


def load(directory, **kw):
    return art.load_artifact(
        directory,
        model_dir=directory,
        encoder_factory=kw.pop("encoder_factory", lambda _d: Encoder()),
        session_factory=fake_session_factory,
        tokenizer_factory=fake_tokenizer_factory,
        **kw,
    )


def test_load_artifact_ok_and_degraded(tmp_path):
    write_artifact_dir(tmp_path)
    a = load(tmp_path)
    assert a.t1 is not None
    assert a.t2 is not None
    assert a.warnings == []
    assert a.reference.n_decks == 3

    no_enc = load(tmp_path, encoder_factory=lambda _d: None)
    assert no_enc.t2 is None
    assert any("T2" in w for w in no_enc.warnings)

    (tmp_path / "t1_int8.onnx").unlink()
    only_t2 = load(tmp_path)
    assert only_t2.t1 is None
    assert any("T1" in w for w in only_t2.warnings)

    (tmp_path / "t2.npz").unlink()
    none = load(tmp_path)
    assert none.t1 is None
    assert none.t2 is None
    assert any("No text model" in w for w in none.warnings)


def test_load_artifact_errors(tmp_path):
    write_artifact_dir(tmp_path)
    (tmp_path / "text_tables.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ScoringError):
        load(tmp_path)
    (tmp_path / "reference.json").write_text(
        '{"format":1,"n_decks":1,"n_concepts":1,"counts":{},"tables":{},"feature_schema_hash":"x"}',
        encoding="utf-8",
    )
    with pytest.raises(ScoringError):
        load(tmp_path)  # schema hash mismatch


def test_get_artifact_caches(tmp_path):
    write_artifact_dir(tmp_path)
    art.clear_artifact_cache()
    kw = {
        "encoder_factory": lambda _d: Encoder(),
        "session_factory": fake_session_factory,
        "tokenizer_factory": fake_tokenizer_factory,
    }
    first = art.get_artifact(tmp_path, model_dir=tmp_path, **kw)
    assert art.get_artifact(tmp_path, model_dir=tmp_path, **kw) is first
    art.clear_artifact_cache()
    assert art.get_artifact(tmp_path, model_dir=tmp_path, **kw) is not first
    art.clear_artifact_cache()
