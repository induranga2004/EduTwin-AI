import pytest

from edusync_c1.schemas import Page, Transcript, TranscriptSegment
from edusync_c1.semantic import SemanticMatcher
from edusync_c1.timeline import build_timeline_compatibility, normalize_transcription_result

LECTURE_LINES = [
    "Gradient Descent minimizes a Loss Function.",
    "Gradient Descent uses a Learning Rate to update weights.",
]


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_normalize_dict_and_fill_text():
    result = normalize_transcription_result(
        {
            "language": "en",
            "segments": [
                {"start": "1.5", "end": None, "text": " hi "},
                {"id": 7, "start": 2, "end": 3, "text": "there"},
            ],
        }
    )
    assert result.text == "hi there"
    assert result.language == "en"
    assert [s.id for s in result.segments] == [0, 7]
    assert result.segments[0].start == 1.5
    assert result.segments[0].end == 0.0


def test_normalize_models_and_objects():
    transcript = Transcript(segments=[TranscriptSegment(id=1, start=0, end=1, text="x")], text="t")
    assert normalize_transcription_result(transcript).text == "t"
    obj = _Obj(text="", language=None, segments=[_Obj(id=3, start=0, end=2, text="obj seg")])
    result = normalize_transcription_result(obj)
    assert result.segments[0].id == 3
    assert result.text == "obj seg"
    assert normalize_transcription_result({}).segments == []


def test_normalize_rejects_unsupported():
    with pytest.raises(ValueError, match="unsupported"):
        normalize_transcription_result(42)


def test_build_timeline_accumulates_evidence():
    pages = [Page(page=1, text=" ".join(LECTURE_LINES))]
    transcript = Transcript(
        segments=[
            TranscriptSegment(id=0, start=0, end=3, text="nothing relevant"),
            TranscriptSegment(id=1, start=3, end=6, text="gradient descent works"),
            TranscriptSegment(id=2, start=6, end=0, text="Gradient Descent again"),
        ]
    )
    result = build_timeline_compatibility(transcript, pages, SemanticMatcher(enabled=False))
    steps = result.timeline
    assert len(steps) == 3
    assert all(n.status == "not_yet_observed" for n in result.initial_graph.nodes)

    def node(step):
        return next(n for n in step.nodes if n.id == "gradient descent")

    assert node(steps[0]).status == "not_yet_observed"
    assert node(steps[1]).status == "introduced"
    assert node(steps[1]).latest_similarity == 1.0
    assert node(steps[2]).evidence_count == 2
    assert steps[1].matched_concepts == ["Gradient Descent"]
    assert steps[2].end_seconds == 6  # zero end falls back to start
    assert steps[1].matching_method == "exact_phrase_fallback"
    assert result.matching.similarity_threshold == 0.38
    assert "Mention does not prove" in result.note


def test_default_matcher_used_when_none_given():
    pages = [Page(page=1, text="Alpha Beta.")]
    result = build_timeline_compatibility(Transcript(segments=[]), pages)
    assert result.timeline == []
