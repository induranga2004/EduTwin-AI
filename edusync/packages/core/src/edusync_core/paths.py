"""Data directory layout under ``CoreSettings.data_dir``."""

from dataclasses import dataclass
from pathlib import Path

from edusync_core.settings import CoreSettings, get_settings


@dataclass(frozen=True)
class DataDirs:
    """Resolved on-disk locations for EduSync data."""

    root: Path
    uploads: Path
    processed: Path
    models: Path
    recordings: Path


def ensure_data_dirs(settings: CoreSettings | None = None) -> DataDirs:
    """Create (if missing) and return the data directories."""
    root = (settings or get_settings()).data_dir
    dirs = DataDirs(
        root=root,
        uploads=root / "uploads",
        processed=root / "processed",
        models=root / "models",
        recordings=root / "recordings",
    )
    for path in (dirs.root, dirs.uploads, dirs.processed, dirs.models, dirs.recordings):
        path.mkdir(parents=True, exist_ok=True)
    return dirs
