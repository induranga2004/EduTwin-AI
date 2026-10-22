from pathlib import Path

import pytest

from edusync_blueprint_model.config import load_config
from edusync_blueprint_model.download import (
    BASE_PATTERNS,
    REQUIRED_FILES,
    download_base,
    verify_base_files,
)


def test_download_with_fake_downloader_and_idempotent(config_file: Path) -> None:
    cfg = load_config(config_file)
    calls: list[dict] = []

    def fake(**kwargs) -> str:
        calls.append(kwargs)
        for name in REQUIRED_FILES:
            (Path(kwargs["local_dir"]) / name).write_text("x")
        return kwargs["local_dir"]

    target = download_base(cfg, downloader=fake)
    assert target == cfg.paths.artifacts_dir / "base" / "org--tiny-model"
    assert calls[0]["repo_id"] == "org/tiny-model"
    assert calls[0]["revision"] == "abc1234567"
    assert calls[0]["allow_patterns"] == BASE_PATTERNS
    assert download_base(cfg, downloader=fake) == target
    assert len(calls) == 1


def test_download_incomplete_raises(config_file: Path) -> None:
    cfg = load_config(config_file)
    with pytest.raises(FileNotFoundError, match="missing"):
        download_base(cfg, downloader=lambda **kw: None)


def test_verify_base_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        verify_base_files(tmp_path)
