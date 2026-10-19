from edusync_c1.coverage import CoverageTracker
from edusync_c1.schemas import ConceptNode, KnowledgeGraph, TranscriptSegment
from edusync_core.models import EvidenceSource


def _graph() -> KnowledgeGraph:
    nodes = [
        ConceptNode(id="gradient descent", label="Gradient Descent"),
        ConceptNode(id="loss", label="Loss"),
        ConceptNode(id="", label="skipped"),
        ConceptNode(id="blank", label="  "),
    ]
    return KnowledgeGraph(node_count=4, edge_count=0, nodes=nodes, edges=[], note="n")


def _seg(i: int, start: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=i, start=start, end=start + 1, text=text)


def test_status_progression_and_ordering():
    tracker = CoverageTracker(_graph())
    snaps = tracker.process_segments(
        [
            _seg(2, 20, "gradient descent once more, and loss"),
            _seg(0, 0, "Intro without concepts"),
            _seg(1, 10, "Now GRADIENT DESCENT."),
            _seg(3, 30, "   "),
        ]
    )
    assert [s.segment_id for s in snaps] == [0, 1, 2, 3]
    by_id = {n.id: n for n in snaps[0].nodes}
    assert set(by_id) == {"gradient descent", "loss"}
    assert by_id["loss"].status == "not_yet_observed"
    assert snaps[1].matched_concepts == ["gradient descent"]
    assert snaps[1].nodes[0].status == "introduced"
    last = snaps[2].nodes[0]
    assert last.status == "repeatedly_mentioned"
    assert (last.first_seen_seconds, last.latest_seen_seconds) == (10, 20)
    assert snaps[3].matched_concepts == []


def test_word_boundaries():
    tracker = CoverageTracker(_graph())
    assert tracker.process_segment(_seg(0, 0, "lossless coding")).matched_concepts == []


def test_evidence_items_are_speech_evidence():
    tracker = CoverageTracker(_graph())
    tracker.process_segments([_seg(0, 5, "loss and gradient descent"), _seg(1, 2, "loss")])
    items = tracker.evidence_items("meeting-1")
    assert [i.timestamp.seconds for i in items] == [2, 5, 5]
    assert all(i.source is EvidenceSource.SPEECH for i in items)
    assert items[0].timestamp.meeting_id == "meeting-1"
    assert items[0].concept.label == "Loss"
    assert items[0].meta["segment_id"] == 1
