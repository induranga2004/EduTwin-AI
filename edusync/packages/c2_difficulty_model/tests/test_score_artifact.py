"""The score artifact builder: files, manifest, loadability and parity with the trainer's T2."""

import json

import numpy as np
import pytest
from t1_fakes import make_examples
from test_t2 import corpus, fake_embed

from edusync_c2.difficulty.scoring.artifact import load_artifact
from edusync_c2.difficulty.scoring.text_models import T2Scorer
from edusync_c2_model import score_artifact as sa
from edusync_c2_model.extract import extract_lecture
from edusync_c2_model.public_data.splits import make_group_split, write_split_once
from edusync_c2_model.t2 import T2Model


class Session:
    def run(self, _outputs, feeds):
        return [feeds["input_ids"].sum(axis=1, keepdims=True).astype(np.float32) / 500.0]


def session_factory(_path):
    return Session()


def tokenizer_factory(_path, max_length):
    return lambda texts: [[1, *[ord(c) % 40 for c in t[:max_length]], 2] for t in texts]


class Encoder:
    encode = staticmethod(fake_embed)


@pytest.fixture
def model_dirs(tmp_path):
    texts, targets = corpus(30)
    model = T2Model(fake_embed, "abcd1234", 20).fit(texts, targets, seed=1)
    t2_dir = tmp_path / "t2"
    model.save(t2_dir)
    t1_dir = tmp_path / "t1"
    t1_dir.mkdir()
    for name in ("model_int8.onnx", "model_fp32.onnx"):
        (t1_dir / name).write_bytes(b"onnx")
    (t1_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (t1_dir / "inference_config.json").write_text('{"max_length": 12}', encoding="utf-8")
    return model, t1_dir, t2_dir


def test_t2_parity_with_trainer(model_dirs, tmp_path):
    model, _, t2_dir = model_dirs
    scorer = T2Scorer.load(t2_dir, fake_embed, [0.0, 1.0])
    texts, _ = corpus(8)
    assert np.allclose(scorer.z(texts), model.predict(texts)[0])


def test_build_artifact(model_dirs, make_entry, tmp_path):
    _, t1_dir, t2_dir = model_dirs
    rows = extract_lecture(make_entry(7, lecture_id="L1", graph=True)) + extract_lecture(
        make_entry(9, lecture_id="L2", graph=False)
    )
    assert any(s != "computed" for r in rows for s in r.statuses)
    out = tmp_path / "score"
    texts, _ = corpus(25)
    manifest = sa.build_artifact(
        rows=rows,
        train_texts=texts,
        t1_dir=t1_dir,
        t2_dir=t2_dir,
        out_dir=out,
        embed=fake_embed,
        session_factory=session_factory,
        tokenizer_factory=tokenizer_factory,
        include_fp32=True,
        train_note="test",
    )
    assert manifest["n_decks"] == 2
    assert manifest["provisional"]
    assert sum(manifest["weights"].values()) == pytest.approx(1.0)
    for name in (
        "reference.json",
        "combiner.json",
        "text_tables.json",
        "t1_fp32.onnx",
        "model_card.md",
    ):
        assert (out / name).is_file()
    stored = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    assert stored["files"]["t2.npz"] == sa.sha256_file(out / "t2.npz")
    assert "provisional" in (out / "model_card.md").read_text(encoding="utf-8")
    loaded = load_artifact(
        out,
        model_dir=out,
        encoder_factory=lambda _d: Encoder(),
        session_factory=session_factory,
        tokenizer_factory=tokenizer_factory,
    )
    assert loaded.t1 is not None
    assert loaded.t2 is not None
    assert loaded.warnings == []


def test_no_fp32_by_default(model_dirs, tmp_path):
    _, t1_dir, t2_dir = model_dirs
    names = sa.copy_models(t1_dir, t2_dir, tmp_path / "o", include_fp32=False)
    assert "t1_fp32.onnx" not in names
    assert "t1_int8.onnx" in names


def test_clear_train_texts(tmp_path):
    examples = make_examples("clear", 20)
    split = make_group_split({e.group_id for e in examples}, seed=1, source="clear")
    (tmp_path / "splits").mkdir()
    write_split_once(tmp_path / "splits" / "clear_split.json", split)
    all_texts = sa.clear_train_texts(tmp_path, None, loader=lambda _p: examples)
    assert 0 < len(all_texts) < 20
    assert sa.clear_train_texts(tmp_path, 2, loader=lambda _p: examples) == all_texts[:2]
