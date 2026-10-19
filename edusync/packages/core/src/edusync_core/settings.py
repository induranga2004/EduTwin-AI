"""Runtime configuration shared by all components (env prefix ``EDUSYNC_``)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUSYNC_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    max_upload_mb: int = 200

    # BigBlueButton integration (empty = adapter disabled)
    bbb_url: str = ""
    bbb_secret: str = ""
    hasura_url: str = ""
    hasura_admin_secret: str = ""
    redis_url: str = ""

    # Speech-to-text
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"


@lru_cache
def get_settings() -> CoreSettings:
    return CoreSettings()
