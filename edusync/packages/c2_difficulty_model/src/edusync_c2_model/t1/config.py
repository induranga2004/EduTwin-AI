"""Training configuration for T1 (a frozen dataclass; unknown keys are rejected)."""

from dataclasses import asdict, dataclass, fields

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
LOSSES = ("huber", "mse")


class ConfigError(ValueError):
    """The training configuration is invalid."""


@dataclass(frozen=True)
class T1Config:
    max_length: int = 256
    epochs: int = 3
    batch_size: int = 16
    lr_encoder: float = 3e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    loss: str = "huber"
    huber_delta: float = 1.0
    seed: int = 42
    threads: int = 4
    freeze_encoder: bool = False
    limit_rows: int | None = None
    ship_threshold: float = 0.02

    def __post_init__(self) -> None:
        positive = ("max_length", "epochs", "batch_size", "threads")
        for name in positive:
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be at least 1")
        for name in ("lr_encoder", "lr_head", "huber_delta", "ship_threshold"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")
        if not 0 <= self.warmup_ratio < 1:
            raise ConfigError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.dropout < 1:
            raise ConfigError("dropout must be in [0, 1)")
        if self.loss not in LOSSES:
            raise ConfigError(f"loss must be one of {LOSSES}")
        if self.limit_rows is not None and self.limit_rows < 1:
            raise ConfigError("limit_rows must be at least 1 when set")

    @classmethod
    def from_dict(cls, data: dict) -> "T1Config":
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ConfigError(f"unknown config keys: {unknown}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
