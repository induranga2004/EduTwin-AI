"""C2 settings, read from ``EDUSYNC_C2_*`` environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class C2Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDUSYNC_C2_", env_file=".env", extra="ignore")

    log_level: str = "INFO"
    model: Literal["baseline", "minilm"] = "baseline"
    model_dir: Path | None = None
    concept_filter: Literal["mock_keep_all", "heuristic"] = "mock_keep_all"
    concept_filter_extras: bool = False
    graph_provider: Literal["mock", "file", "c1_api"] = "mock"
    graph_file: Path | None = None
    c1_base_url: str = "http://127.0.0.1:8100/api/c1"
    c1_timeout_s: float = Field(default=60, gt=0)
    semantic: Literal["off", "minilm"] = "off"
    semantic_model_dir: Path | None = None
    semantic_def_threshold: float = Field(default=0.23, ge=0, le=1)
    semantic_ex_threshold: float = Field(default=0.23, ge=0, le=1)
    semantic_max_sentences: int = Field(default=400, ge=1)

    score_dir: Path | None = None  # layered-analysis artifact folder (see difficulty/scoring)
    demo_graph_dir: Path | None = (
        None  # hand-made graphs: <deck stem>.graph.json, default.graph.json
    )

    @field_validator(
        "model_dir",
        "graph_file",
        "semantic_model_dir",
        "score_dir",
        "demo_graph_dir",
        mode="before",
    )
    @classmethod
    def _empty_dir_is_none(cls, value: object) -> object:
        return None if value == "" else value

    @field_validator("c1_base_url")
    @classmethod
    def _check_base_url(cls, value: str) -> str:
        url = value.strip()
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("c1_base_url must start with http:// or https://")
        return url

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
