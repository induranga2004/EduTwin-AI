"""C2 settings, read from ``EDUSYNC_C2_*`` environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class C2Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUSYNC_C2_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    model: Literal["baseline", "minilm"] = "baseline"
    model_dir: Path | None = None

    @field_validator("model_dir", mode="before")
    @classmethod
    def _empty_dir_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("log_level")
    @classmethod
    def _check_level(cls, value: str) -> str:
        level = value.strip().upper()
        if level not in _LEVELS:
            raise ValueError(f"log_level must be one of {', '.join(_LEVELS)}")
        return level


@lru_cache
def get_c2_settings() -> C2Settings:
    return C2Settings()
