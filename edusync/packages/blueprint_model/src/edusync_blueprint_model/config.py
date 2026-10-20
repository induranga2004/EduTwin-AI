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
    path: Path
    text_column: str = Field(min_length=1)
    label_column: str = Field(min_length=1)
    val_fraction: float = Field(ge=0.0, lt=1.0)
    test_fraction: float = Field(ge=0.0, lt=1.0)
    seed: int = 42

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
    cfg.data.path = _resolve(cfg.data.path, folder)
    cfg.paths.artifacts_dir = _resolve(cfg.paths.artifacts_dir, folder)
    return cfg
