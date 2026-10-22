from pathlib import Path

import pytest
from pydantic import ValidationError

from edusync_c2.settings import C2Settings, get_c2_settings


def test_defaults(monkeypatch):
    for k in ("LOG_LEVEL", "MODEL", "MODEL_DIR"):
        monkeypatch.delenv(f"EDUSYNC_C2_{k}", raising=False)
    s = C2Settings(_env_file=None)
    assert (s.log_level, s.model, s.model_dir) == ("INFO", "baseline", None)


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_LOG_LEVEL", " debug ")
    monkeypatch.setenv("EDUSYNC_C2_MODEL", "minilm")
    monkeypatch.setenv("EDUSYNC_C2_MODEL_DIR", "some/dir")
    s = C2Settings(_env_file=None)
    assert s.log_level == "DEBUG"
    assert s.model == "minilm"
    assert s.model_dir == Path("some/dir")


def test_empty_model_dir_is_none(monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_MODEL_DIR", "")
    assert C2Settings(_env_file=None).model_dir is None


def test_invalid_level_and_model(monkeypatch):
    with pytest.raises(ValidationError, match="log_level"):
        C2Settings(log_level="TRACE", _env_file=None)
    with pytest.raises(ValidationError):
        C2Settings(model="gpt", _env_file=None)


def test_unrelated_env_is_ignored(monkeypatch):
    monkeypatch.setenv("EDUSYNC_C2_UNKNOWN", "x")
    C2Settings(_env_file=None)


def test_get_c2_settings_is_cached(monkeypatch):
    get_c2_settings.cache_clear()
    assert get_c2_settings() is get_c2_settings()
    get_c2_settings.cache_clear()
