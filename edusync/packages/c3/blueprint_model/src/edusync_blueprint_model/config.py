"""Blueprint configuration: typed models, TOML loading and artifact paths."""

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config" / "blueprint.toml"


class ConfigError(ValueError):
    """The configuration file is missing or invalid."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Section):
    name: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    max_length: int = Field(ge=2, le=512)


class DataConfig(_Section):
    # Either one CSV (`path`, split by the fractions below) or three pre-split CSVs.
    path: Path | None = None
    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None
    text_column: str = Field(min_length=1)
    label_column: str = Field(min_length=1)
    val_fraction: float = Field(ge=0.0, lt=1.0, default=0.2)
    test_fraction: float = Field(ge=0.0, lt=1.0, default=0.2)
    seed: int = 42

    @property
    def presplit(self) -> bool:
        return self.train_path is not None

    @model_validator(mode="after")
    def _paths_are_consistent(self) -> "DataConfig":
        split_paths = (self.train_path, self.val_path, self.test_path)
        if any(split_paths) and not all(split_paths):
            raise ValueError("train_path, val_path and test_path must be set together")
        if all(split_paths) and self.path is not None:
            raise ValueError("set either path or train_path/val_path/test_path, not both")
        if not any(split_paths) and self.path is None:
            raise ValueError("set path, or train_path/val_path/test_path")
        return self

    @model_validator(mode="after")
    def _fractions_leave_room_for_training(self) -> "DataConfig":
        if self.val_fraction + self.test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be below 1.0")
        return self


class TrainConfig(_Section):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    encoder_lr: float = Field(gt=0)
    head_lr: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    dropout: float = Field(ge=0.0, lt=1.0)
    threads: int = Field(ge=0, default=0)
    seed: int = 42


class QuantizeConfig(_Section):
    weight_type: Literal["qint8", "quint8"] = "qint8"
    per_channel: bool = False
    op_types: list[str] = Field(default_factory=lambda: ["MatMul"], min_length=1)
    benchmark_runs: int = Field(ge=1, default=50)


class DecisionConfig(_Section):
    """Optional binary decision rule: predict ``positive_label`` when its probability >= threshold.

    Unset (the default) means plain argmax. A threshold below 0.5 trades precision for recall
    on the positive label.
    """

    positive_label: str | None = None
    threshold: float = Field(gt=0.0, lt=1.0, default=0.5)


class PathsConfig(_Section):
    artifacts_dir: Path


class ArtifactPaths:
    """Where each pipeline stage reads and writes its files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def base_dir(self) -> Path:
        return self.root / "base"

    @property
    def finetuned_dir(self) -> Path:
        return self.root / "finetuned"

    @property
    def onnx_dir(self) -> Path:
        return self.root / "onnx"

    @property
    def quantized_dir(self) -> Path:
        return self.root / "quantized"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def base_model_dir(self, model_name: str) -> Path:
        return self.base_dir / model_name.replace("/", "--")


class BlueprintConfig(_Section):
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    quantize: QuantizeConfig
    paths: PathsConfig
    decision: DecisionConfig = Field(default_factory=DecisionConfig)
    config_dir: Path = Path()

    @property
    def artifacts(self) -> ArtifactPaths:
        return ArtifactPaths(self.paths.artifacts_dir)


def _resolve(value: Path, base: Path) -> Path:
    return value if value.is_absolute() else (base / value).resolve()


def load_config(path: str | Path | None = None) -> BlueprintConfig:
    """Load and validate a blueprint TOML file (default: the packaged one)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise ConfigError(f"Config file not found: {cfg_path}")
    try:
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {cfg_path}: {exc}") from exc
    folder = cfg_path.resolve().parent
    try:
        cfg = BlueprintConfig(**raw, config_dir=folder)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        raise ConfigError(f"Invalid config {cfg_path}: {problems}") from exc
    for name in ("path", "train_path", "val_path", "test_path"):
        value = getattr(cfg.data, name)
        if value is not None:
            setattr(cfg.data, name, _resolve(value, folder))
    cfg.paths.artifacts_dir = _resolve(cfg.paths.artifacts_dir, folder)
    return cfg
