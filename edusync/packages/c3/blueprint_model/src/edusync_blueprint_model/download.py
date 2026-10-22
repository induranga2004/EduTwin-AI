"""Fetch the pinned base model from Hugging Face."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from edusync_blueprint_model.config import BlueprintConfig
from edusync_blueprint_model.deps import require

logger = logging.getLogger(__name__)

BASE_PATTERNS = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "modules.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
]
REQUIRED_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def verify_base_files(directory: Path) -> None:
    """Raise FileNotFoundError listing required files missing from ``directory``."""
    missing = [f for f in REQUIRED_FILES if not (Path(directory) / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Base model files missing in {directory}: {missing}")


def _default_downloader(**kwargs: Any) -> Any:  # pragma: no cover - network
    return require("huggingface_hub", "download").snapshot_download(**kwargs)


def download_base(config: BlueprintConfig, *, downloader: Callable[..., Any] | None = None) -> Path:
    """Download the base model into the artifacts folder (skipped if already complete)."""
    target = config.artifacts.base_model_dir(config.model.name)
    try:
        verify_base_files(target)
    except FileNotFoundError:
        pass
    else:
        logger.info("Base model already present at %s", target)
        return target
    target.mkdir(parents=True, exist_ok=True)
    (downloader or _default_downloader)(
        repo_id=config.model.name,
        revision=config.model.revision,
        allow_patterns=BASE_PATTERNS,
        local_dir=str(target),
    )
    verify_base_files(target)
    return target
