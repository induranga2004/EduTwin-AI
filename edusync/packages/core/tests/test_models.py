import pytest
from pydantic import ValidationError

from edusync_core import (
    ConceptRef,
    EvidenceItem,
    EvidenceSource,
    Flag,
    FlagSeverity,
    LectureTimestamp,
)


def _ts(s: float = 1.0) -> LectureTimestamp:
    return LectureTimestamp(meeting_id="m1", seconds=s)


def test_timestamp_rejects_negative():
    with pytest.raises(ValidationError):
        _ts(-1)


def test_evidence_defaults_and_bounds():
    item = EvidenceItem(
        concept=ConceptRef(id="c", label="C"), source=EvidenceSource.SLIDE, timestamp=_ts()
    )
    assert item.confidence == 1.0
    assert item.meta == {}
    with pytest.raises(ValidationError):
        EvidenceItem(
            concept=item.concept, source=EvidenceSource.CHAT, timestamp=_ts(), confidence=2
        )


def test_flag_defaults():
    flag = Flag(concept=ConceptRef(id="c", label="C"), timestamp=_ts(), component="c1", reason="r")
    assert flag.severity is FlagSeverity.REVIEW
    assert flag.evidence == []
