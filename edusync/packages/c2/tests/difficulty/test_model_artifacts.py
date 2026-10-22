"""Artifact-folder validation of OnnxFusionModel and the model factory."""

import json
from pathlib import Path

import pytest

from edusync_c2.difficulty.errors import ModelArtifactError, ModelNotAvailableError
from edusync_c2.difficulty.models import (
    ARTIFACT_SPEC,
    OnnxFusionModel,
    RuleBaselineModel,
    create_model,
)
from edusync_c2.difficulty.registry import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_schema_hash,
)
from edusync_c2.difficulty.schemas import ModelKind


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def make_artifacts(
    root: Path, *, int8: bool = True, fp32: bool = False, trained_on: str | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if int8:
        (root / "model_int8.onnx").write_bytes(b"int8")
    if fp32:
        (root / "model_fp32.onnx").write_bytes(b"fp32")
    write_json(root / "labels.json", ["Low", "Medium", "High"])
    write_json(
        root / "feature_schema.json",
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "names": list(FEATURE_NAMES),
            "hash": feature_schema_hash(),
        },
    )
    config = {"max_length": 256, "base_model": "all-MiniLM-L6-v2", "base_revision": "abc123"}
    if trained_on is not None:
        config["trained_on"] = trained_on
    write_json(root / "inference_config.json", config)
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


@pytest.fixture
def model_dir(tmp_path) -> Path:
    return make_artifacts(tmp_path / "model")


def patch_json(model_dir: Path, name: str, **changes) -> None:
    path = model_dir / name
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    write_json(path, data)


def test_valid_int8_only(model_dir):
    model = OnnxFusionModel.from_artifacts(model_dir)
    assert model.onnx_file.name == "model_int8.onnx"
    assert model.info.name == "onnx_fusion"
    assert model.info.version == FEATURE_SCHEMA_VERSION
    assert model.info.kind is ModelKind.TODO
    assert model.info.trained_on is None
    assert "inference is not implemented" in model.info.description


def test_valid_fp32_only(tmp_path):
    model = OnnxFusionModel.from_artifacts(make_artifacts(tmp_path, int8=False, fp32=True))
    assert model.onnx_file.name == "model_fp32.onnx"


def test_int8_preferred_when_both(tmp_path):
    model = OnnxFusionModel.from_artifacts(make_artifacts(tmp_path, int8=True, fp32=True))
    assert model.onnx_file.name == "model_int8.onnx"


def test_trained_on_is_read(tmp_path):
    root = make_artifacts(tmp_path, trained_on="public graded text 2026")
    assert OnnxFusionModel.from_artifacts(root).info.trained_on == "public graded text 2026"


def test_predict_not_implemented(model_dir):
    with pytest.raises(NotImplementedError, match="Phase 3"):
        OnnxFusionModel.from_artifacts(model_dir).predict([])


def test_missing_folder(tmp_path):
    with pytest.raises(ModelArtifactError, match="does not exist"):
        OnnxFusionModel.from_artifacts(tmp_path / "nope")


def test_folder_is_a_file(tmp_path):
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(ModelArtifactError, match="does not exist"):
        OnnxFusionModel.from_artifacts(file)


def test_no_onnx_file(tmp_path):
    root = make_artifacts(tmp_path, int8=False, fp32=False)
    with pytest.raises(ModelArtifactError, match=r"model_int8\.onnx or model_fp32\.onnx"):
        OnnxFusionModel.from_artifacts(root)


@pytest.mark.parametrize("name", ["labels.json", "feature_schema.json", "inference_config.json"])
def test_missing_json_file(model_dir, name):
    (model_dir / name).unlink()
    with pytest.raises(ModelArtifactError, match=f"missing {name}"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_missing_tokenizer(model_dir):
    (model_dir / "tokenizer.json").unlink()
    with pytest.raises(ModelArtifactError, match=r"missing tokenizer.json"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_empty_tokenizer(model_dir):
    (model_dir / "tokenizer.json").write_bytes(b"")
    with pytest.raises(ModelArtifactError, match=r"tokenizer.json is empty"):
        OnnxFusionModel.from_artifacts(model_dir)


@pytest.mark.parametrize(
    "labels",
    [["High", "Low", "Medium"], ["Low", "Medium"], ["High", "Medium", "Low"], {"labels": []}],
)
def test_wrong_labels(model_dir, labels):
    write_json(model_dir / "labels.json", labels)
    with pytest.raises(ModelArtifactError, match="ordinal order"):
        OnnxFusionModel.from_artifacts(model_dir)


@pytest.mark.parametrize("name", ["labels.json", "feature_schema.json", "inference_config.json"])
def test_invalid_json(model_dir, name):
    (model_dir / name).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelArtifactError, match=f"{name} is not readable valid JSON"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_schema_hash_mismatch(model_dir):
    patch_json(model_dir, "feature_schema.json", hash="0" * 64)
    with pytest.raises(ModelArtifactError, match="registry changed since training"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_schema_names_order_mismatch(model_dir):
    names = list(FEATURE_NAMES)
    names[0], names[1] = names[1], names[0]
    patch_json(model_dir, "feature_schema.json", names=names)
    with pytest.raises(ModelArtifactError, match="registry changed since training"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_schema_missing_names(model_dir):
    write_json(
        model_dir / "feature_schema.json",
        {"schema_version": FEATURE_SCHEMA_VERSION, "hash": feature_schema_hash()},
    )
    with pytest.raises(ModelArtifactError, match="names do not match"):
        OnnxFusionModel.from_artifacts(model_dir)


@pytest.mark.parametrize("schema", [["not", "an", "object"], {"names": []}, {"schema_version": ""}])
def test_schema_without_version(model_dir, schema):
    write_json(model_dir / "feature_schema.json", schema)
    with pytest.raises(ModelArtifactError, match="schema_version"):
        OnnxFusionModel.from_artifacts(model_dir)


@pytest.mark.parametrize("value", [0, -1, 513, "256", 256.0, None, True])
def test_bad_max_length(model_dir, value):
    patch_json(model_dir, "inference_config.json", max_length=value)
    with pytest.raises(ModelArtifactError, match="max_length"):
        OnnxFusionModel.from_artifacts(model_dir)


@pytest.mark.parametrize("key", ["base_model", "base_revision"])
@pytest.mark.parametrize("value", [None, "", "  ", 3])
def test_bad_base_fields(model_dir, key, value):
    patch_json(model_dir, "inference_config.json", **{key: value})
    with pytest.raises(ModelArtifactError, match=key):
        OnnxFusionModel.from_artifacts(model_dir)


def test_bad_trained_on(model_dir):
    patch_json(model_dir, "inference_config.json", trained_on=5)
    with pytest.raises(ModelArtifactError, match="trained_on"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_inference_config_not_object(model_dir):
    write_json(model_dir / "inference_config.json", [1])
    with pytest.raises(ModelArtifactError, match="must be a JSON object"):
        OnnxFusionModel.from_artifacts(model_dir)


def test_max_length_boundaries(model_dir):
    for ok in (1, 512):
        patch_json(model_dir, "inference_config.json", max_length=ok)
        OnnxFusionModel.from_artifacts(model_dir)


def test_spec_text_mentions_every_file():
    for token in (
        "model_int8.onnx",
        "model_fp32.onnx",
        "labels.json",
        "feature_schema.json",
        "inference_config.json",
        "tokenizer.json",
    ):
        assert token in ARTIFACT_SPEC


# ------------------------------------------------------------------------------ create_model


def test_create_baseline_default():
    assert isinstance(create_model(), RuleBaselineModel)
    assert isinstance(create_model("baseline", Path("ignored")), RuleBaselineModel)


def test_create_minilm_without_dir():
    with pytest.raises(ModelNotAvailableError, match="EDUSYNC_C2_MODEL_DIR"):
        create_model("minilm")


def test_create_minilm_with_valid_dir(model_dir):
    model = create_model("minilm", model_dir)
    assert isinstance(model, OnnxFusionModel)
    with pytest.raises(NotImplementedError):
        model.predict([])


def test_create_minilm_with_invalid_dir(tmp_path):
    with pytest.raises(ModelArtifactError):
        create_model("minilm", tmp_path / "missing")


def test_create_unknown_name():
    with pytest.raises(ModelNotAvailableError, match="baseline, minilm"):
        create_model("gpt")


def test_schema_version_mismatch_is_checked_first(model_dir):
    patch_json(model_dir, "feature_schema.json", schema_version="0.0", names=["x"], hash="bad")
    with pytest.raises(ModelArtifactError, match="registry changed since training") as info:
        OnnxFusionModel.from_artifacts(model_dir)
    assert "schema_version '0.0'" in str(info.value)
    assert repr(FEATURE_SCHEMA_VERSION) in str(info.value)
