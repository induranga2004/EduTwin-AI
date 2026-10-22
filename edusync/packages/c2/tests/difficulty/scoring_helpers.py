"""Shared fakes for the scoring tests: reference rows, fake text scorers and a full artifact dir."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from edusync_c2.difficulty import DifficultyPipeline
from edusync_c2.difficulty.fusion import to_vector
from edusync_c2.difficulty.providers import PayloadGraphProvider
from edusync_c2.difficulty.registry import FEATURE_SCHEMA_VERSION, feature_schema_hash
from edusync_c2.difficulty.schemas import C1GraphResponse
from edusync_c2.difficulty.scoring.analyze import RecordingModel
from edusync_c2.difficulty.scoring.artifact import ScoreArtifact
from edusync_c2.difficulty.scoring.combiner import fit_combiner
from edusync_c2.difficulty.scoring.reference import build_reference
from edusync_c2.difficulty.scoring.signals import group_scores, score_features
from edusync_c2.difficulty.synthetic import build_c1_graph, build_pptx, make_deck_spec

EMB_DIMS = 4


def deck(seed: int = 5, n: int = 6, lecture: str = "lec-s"):
    spec = make_deck_spec(seed, n, lecture)
    return spec, build_pptx(spec), build_c1_graph(spec, "v1")


def reference_rows():
    """Feature rows of three synthetic decks (what the real cache gives for real decks)."""
    rows = []
    for seed, n in ((5, 6), (7, 8), (11, 7)):
        _, data, graph = deck(seed, n, f"ref-{seed}")
        recorder = RecordingModel()
        provider = PayloadGraphProvider(C1GraphResponse.model_validate(graph))
        DifficultyPipeline(model=recorder, graph_provider_factory=lambda d, f, p=provider: p).run(
            data, "d.pptx"
        )
        for fusion in recorder.captured:
            values, mask = to_vector(fusion)
            statuses = tuple(f.status.value for f in fusion.features)
            rows.append(
                SimpleNamespace(
                    lecture_id=f"ref-{seed}",
                    values=tuple(values),
                    missing=tuple(mask),
                    statuses=statuses,
                    text=fusion.text,
                )
            )
    return rows


class FakeScorer:
    """Stands in for T1/T2: percentile from the text length."""

    def __init__(self, name: str, shift: float = 0.0, fail: bool = False) -> None:
        self.name, self.shift, self.fail = name, shift, fail

    def percentiles(self, texts):
        if self.fail:
            raise RuntimeError("boom")
        return [min(1.0, (len(t) % 10) / 10 + self.shift) for t in texts]


def fake_artifact(*, t1=True, t2=True, warnings=None) -> ScoreArtifact:
    rows = reference_rows()
    reference = build_reference(
        rows, schema_version=FEATURE_SCHEMA_VERSION, schema_hash=feature_schema_hash()
    )
    signal_rows = []
    for r in rows:
        from edusync_c2.difficulty.registry import FEATURE_SPECS
        from edusync_c2.difficulty.schemas import FeatureStatus, FeatureValue

        feats = [
            FeatureValue(
                name=s.name,
                group=s.group,
                value=v if st == "computed" else None,
                status=FeatureStatus(st) if st == "computed" else FeatureStatus.UNAVAILABLE,
                source="t",
            )
            for s, v, st in zip(FEATURE_SPECS, r.values, r.statuses, strict=True)
        ]
        g = group_scores(score_features(feats, reference))
        signal_rows.append({**g, "text": (len(r.text) % 10) / 10})
    combiner = fit_combiner(signal_rows)
    return ScoreArtifact(
        reference,
        combiner,
        FakeScorer("t1") if t1 else None,
        FakeScorer("t2", 0.1) if t2 else None,
        {},
        list(warnings or []),
    )


def write_t2(directory: Path, *, chunk_words=100, kind="t2_ridge", dims_ok=True) -> None:
    dims = 5 + EMB_DIMS
    rng = np.random.default_rng(0)
    config = {
        "format_version": 1,
        "model_kind": kind,
        "embedding_dims": EMB_DIMS,
        "chunk_words": chunk_words,
    }
    np.savez(
        directory / "t2.npz",
        config_json=np.array(json.dumps(config)),
        x_mean=rng.normal(size=dims if dims_ok else dims + 1),
        x_std=np.ones(dims),
        coef=rng.normal(size=dims),
        intercept=np.array([0.1]),
        emb_mean=np.zeros(EMB_DIMS),
    )
    (directory / "t2_config.json").write_text(json.dumps(config), encoding="utf-8")


def fake_embed(texts):
    out = np.zeros((len(texts), EMB_DIMS))
    for i, t in enumerate(texts):
        for j, ch in enumerate(t.encode("utf-8")):
            out[i, j % EMB_DIMS] += ch % 7 + 1
    return out


class FakeSession:
    def run(self, _outputs, feeds):
        ids = feeds["input_ids"]
        return [ids.sum(axis=1, keepdims=True).astype(np.float32) / 1000.0]


def fake_session_factory(path):
    return FakeSession()


def fake_tokenizer_factory(path, max_length):
    return lambda texts: [[1, *[ord(c) % 50 for c in t[:max_length]], 2] for t in texts]


def write_t1(directory: Path, *, fp32_only=False, config=True) -> None:
    (directory / ("t1_fp32.onnx" if fp32_only else "t1_int8.onnx")).write_bytes(b"x")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    if config:
        (directory / "inference_config.json").write_text(
            json.dumps({"max_length": 8}), encoding="utf-8"
        )


def write_artifact_dir(directory: Path, *, with_t1=True) -> None:
    """A complete, loadable artifact folder built from the fake artifact's reference/combiner."""
    art = fake_artifact(t1=False, t2=False)
    directory.mkdir(parents=True, exist_ok=True)
    art.reference.save(directory / "reference.json")
    art.combiner.save(directory / "combiner.json")
    (directory / "text_tables.json").write_text(
        json.dumps({"t1_z_table": [0.0, 1.0], "t2_z_table": [-1.0, 0.0, 1.0]}), encoding="utf-8"
    )
    (directory / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    write_t2(directory)
    if with_t1:
        write_t1(directory)
