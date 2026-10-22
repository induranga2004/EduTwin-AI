"""Tests for skip alerts and late-joiner catch-up endpoints in api.py."""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from edusync_c1.api import get_lecture_alerts, get_lecture_catchup
from edusync_c1.storage import processed_dir
from edusync_core.settings import CoreSettings


@pytest.fixture
def mock_session_settings(tmp_path: Path) -> CoreSettings:
    """Fixture providing isolated storage with a simulated IISA lecture session."""
    settings = CoreSettings(data_dir=tmp_path, max_upload_mb=5)
    sample_path = Path("edusync/packages/c1/data/samples/sample_lecture_graph.json")
    sample_data = json.loads(sample_path.read_text(encoding="utf-8"))

    mock_payload = {
        "recording_id": "test-iisa-session-01",
        "filename": "audio.wav",
        "document_filename": "slides.pdf",
        "language": "en",
        "segment_count": 3,
        "transcript": "Today we jump into User Authentication directly. Then Access Control.",
        "segments": [],
        "concept_count": 6,
        "concepts": sample_data["concepts"],
        "graph": sample_data["graph"],
        "graph_timeline": {
            "timeline": [
                {
                    "start_seconds": 60.0,
                    "end_seconds": 120.0,
                    "transcript_text": "Introduction to Information Security.",
                    "matched_concepts": ["Introduction to Information Security"],
                },
                {
                    "start_seconds": 240.0,
                    "end_seconds": 360.0,
                    "transcript_text": "User Authentication requires password hashing.",
                    "matched_concepts": ["User Authentication"],
                },
                {
                    "start_seconds": 450.0,
                    "end_seconds": 550.0,
                    "transcript_text": "Access Control policies grant permissions.",
                    "matched_concepts": ["Access Control"],
                },
            ]
        },
    }

    out_file = processed_dir(settings) / "test-iisa-session-01_transcript.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(mock_payload), encoding="utf-8")
    return settings


def test_get_lecture_alerts_detects_skips(mock_session_settings: CoreSettings):
    report = get_lecture_alerts("test-iisa-session-01", mock_session_settings)
    assert report.total_skip_alerts == 1
    assert report.coverage_ratio == 0.5
    assert report.pedagogical_alignment_score < 1.0

    alert = report.alerts[0]
    assert alert.concept_id == "user_authentication"
    assert alert.severity == "critical"
    assert "hash_functions" in alert.skipped_prerequisites
    assert "symmetric_encryption" in alert.skipped_prerequisites
    assert "without prior delivery" in alert.message


def test_get_lecture_catchup_orders_topologically(mock_session_settings: CoreSettings):
    pack = get_lecture_catchup(
        "test-iisa-session-01", mock_session_settings, join_seconds=300.0
    )
    assert pack.missed_concept_count == 2
    assert pack.estimated_catchup_minutes <= 5.0

    ordered_ids = [c.concept_id for c in pack.ordered_concepts]
    assert ordered_ids == ["intro_infosec", "user_authentication"]
    assert pack.ordered_concepts[0].chain_depth == 0
    assert pack.ordered_concepts[1].chain_depth == 3
    assert len(pack.quiz) == 2
    assert "Which statement correctly defines" in pack.quiz[0].question


def test_get_lecture_endpoints_404_on_missing(mock_session_settings: CoreSettings):
    with pytest.raises(HTTPException) as exc_alerts:
        get_lecture_alerts("non-existent-rec", mock_session_settings)
    assert exc_alerts.value.status_code == 404

    with pytest.raises(HTTPException) as exc_catchup:
        get_lecture_catchup("non-existent-rec", mock_session_settings)
    assert exc_catchup.value.status_code == 404
