from edusync_core import CoreSettings, ensure_data_dirs, get_settings


def test_creates_dirs(tmp_path):
    settings = CoreSettings(data_dir=tmp_path / "d", _env_file=None)
    dirs = ensure_data_dirs(settings)
    for p in (dirs.root, dirs.uploads, dirs.processed, dirs.models, dirs.recordings):
        assert p.is_dir()
    assert dirs.uploads == tmp_path / "d" / "uploads"
    assert ensure_data_dirs(settings) == dirs  # idempotent


def test_default_settings(monkeypatch, tmp_path):
    get_settings.cache_clear()
    monkeypatch.setenv("EDUSYNC_DATA_DIR", str(tmp_path / "x"))
    try:
        assert ensure_data_dirs().root == tmp_path / "x"
    finally:
        get_settings.cache_clear()
