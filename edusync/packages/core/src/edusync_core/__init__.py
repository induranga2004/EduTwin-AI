"""EduSync core: shared models, settings and BigBlueButton adapters."""

from edusync_core.component import Component
from edusync_core.models import (
    ConceptRef,
    EvidenceItem,
    EvidenceSource,
    Flag,
    FlagSeverity,
    LectureTimestamp,
)
from edusync_core.paths import DataDirs, ensure_data_dirs
from edusync_core.settings import CoreSettings, get_settings

__all__ = [
    "Component",
    "ConceptRef",
    "CoreSettings",
    "DataDirs",
    "EvidenceItem",
    "EvidenceSource",
    "Flag",
    "FlagSeverity",
    "LectureTimestamp",
    "ensure_data_dirs",
    "get_settings",
]
