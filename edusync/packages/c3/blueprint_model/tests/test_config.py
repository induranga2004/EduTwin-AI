from pathlib import Path

import pytest

from edusync_blueprint_model.config import (
    DEFAULT_CONFIG_PATH,
    ArtifactPaths,
    ConfigError,
    load_config,
)


def test_default_config_loads() -> None:
    cfg = load_config()
    assert DEFAULT_CONFIG_PATH.is_file()
    assert cfg.model.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.data.path.is_absolute()
    assert cfg.data.path.name == "sample.csv"
    assert cfg.paths.artifacts_dir.name == "artifacts"


def test_relative_paths_resolve_against_config_file(config_file: Path) -> None:
    cfg = load_config(config_file)
    root = config_file.parent.parent
    assert cfg.data.path == (root / "data.csv").resolve()
    assert cfg.paths.artifacts_dir == (root / "out").resolve()
    assert cfg.config_dir == config_file.parent.resolve()


def test_absolute_paths_kept(config_file: Path, tmp_path: Path) -> None:
    target = (tmp_path / "abs").as_posix()
    config_file.write_text(config_file.read_text().replace('"../out"', f'"{target}"'))
    assert load_config(config_file).paths.artifacts_dir == Path(target)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_bad_toml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[model\nname =")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(bad)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("epochs = 1", "epochs = 0", "train.epochs"),
        ("batch_size = 4", "batch_size = 0", "train.batch_size"),
        ("val_fraction = 0.2", "val_fraction = -0.1", "val_fraction"),
        ("val_fraction = 0.2", "val_fraction = 0.9", "must be below 1.0"),
        ('weight_type = "qint8"', 'weight_type = "int4"', "weight_type"),
        ("max_length = 16", "max_length = 0", "max_length"),
    ],
)
def test_invalid_values(config_file: Path, old: str, new: str, message: str) -> None:
    config_file.write_text(config_file.read_text().replace(old, new))
    with pytest.raises(ConfigError, match=message):
        load_config(config_file)


def test_missing_section(config_file: Path) -> None:
    text = config_file.read_text().split("[paths]")[0]
    config_file.write_text(text)
    with pytest.raises(ConfigError, match="paths"):
        load_config(config_file)


def test_artifact_paths(tmp_path: Path) -> None:
    art = ArtifactPaths(tmp_path)
    assert art.base_dir == tmp_path / "base"
    assert art.finetuned_dir == tmp_path / "finetuned"
    assert art.onnx_dir == tmp_path / "onnx"
    assert art.quantized_dir == tmp_path / "quantized"
    assert art.reports_dir == tmp_path / "reports"
    assert art.base_model_dir("a/b") == tmp_path / "base" / "a--b"


def test_config_artifacts_property(config_file: Path) -> None:
    cfg = load_config(config_file)
    assert cfg.artifacts.root == cfg.paths.artifacts_dir


def test_positive_recall_selection_needs_positive_label(config_file: Path) -> None:
    text = config_file.read_text(encoding="utf-8")
    bad = config_file.with_name("bad_select.toml")
    bad.write_text(text.replace("seed = 1\n\n[quantize]", 'seed = 1\nselect_by = "positive_recall"\n\n[quantize]', 1), encoding="utf-8")
    with pytest.raises(ConfigError, match="positive_label"):
        load_config(bad)
