"""Local file storage helpers; directories are derived from ``CoreSettings.data_dir``."""

from pathlib import Path

from edusync_core.settings import CoreSettings


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def c1_dir(settings: CoreSettings, name: str) -> Path:
    """Return ``<data_dir>/c1/<name>``, creating it."""
    return ensure_dir(settings.data_dir / "c1" / name)


def uploads_dir(settings: CoreSettings) -> Path:
    return c1_dir(settings, "uploads")


def recordings_dir(settings: CoreSettings) -> Path:
    return c1_dir(settings, "recordings")


def processed_dir(settings: CoreSettings) -> Path:
    return c1_dir(settings, "processed")
