from pathlib import Path

import pytest

from edusync_core import CoreSettings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults(monkeypatch):
    monkeypatch.delenv("EDUSYNC_DATA_DIR", raising=False)
    s = CoreSettings(_env_file=None)
    assert s.data_dir == Path("data")
    assert s.whisper_model == "base"


def test_env_override_and_cache(monkeypatch):
    monkeypatch.setenv("EDUSYNC_MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("EDUSYNC_BBB_URL", "https://bbb.example")
    s = get_settings()
    assert s.max_upload_mb == 7
    assert s.bbb_url == "https://bbb.example"
    assert get_settings() is s
