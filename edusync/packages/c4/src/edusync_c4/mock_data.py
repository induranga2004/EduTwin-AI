"""MOCK data for developing and testing Phase 4.

Nothing here comes from real BBB recordings or from any other EduTwin component (C1/C2/C3);
it is invented, self-contained sample data.
"""

from edusync_c4.schemas import SlideText, SpeechSegment
from edusync_core import ConceptRef

IS_MOCK = True
MOCK_LECTURE_ID = "mock-lecture-1"
MOCK_SLIDE_ID = "mock-slide-1"

MOCK_CONCEPTS = [
    ConceptRef(id="kinetic-energy", label="Kinetic energy"),
    ConceptRef(id="momentum", label="Momentum"),
    ConceptRef(id="work", label="Work"),
]

MOCK_SLIDE_TEXT = SlideText(
    slide_id=MOCK_SLIDE_ID,
    text=(
        "Energy and motion. Kinetic energy depends on mass and speed. "
        "Momentum is mass times velocity."
    ),
)

MOCK_TRANSCRIPT = [
    SpeechSegment(id=0, start=0.0, end=20.0, text="Welcome, today we look at energy."),
    SpeechSegment(
        id=1, start=100.0, end=118.0, text="Kinetic energy grows with the square of speed."
    ),
    SpeechSegment(id=2, start=120.0, end=140.0, text="Let me write that down on the board."),
    SpeechSegment(id=3, start=400.0, end=420.0, text="Now a different topic, unrelated."),
]

# correct / total quiz tallies per concept id (invented)
MOCK_COMPREHENSION = {"kinetic-energy": (2, 10), "momentum": (9, 10)}

# Raw events shaped like BBB tldraw shapes, timestamps in lecture seconds (invented).
MOCK_ANNOTATION_EVENTS = [
    {
        "id": "shape:t1",
        "type": "text",
        "x": 100,
        "y": 100,
        "timestamp": 120.0,
        "props": {"text": "Kinetic energy", "w": 140, "h": 30},
    },
    {
        "id": "shape:d1",
        "type": "draw",
        "x": 100,
        "y": 140,
        "timestamp": 125.0,
        "props": {"segments": [{"points": [[0, 0, 0.5], [60, 5, 0.5], [120, 0, 0.5]]}]},
    },
    {
        "id": "shape:far",
        "type": "text",
        "x": 900,
        "y": 700,
        "timestamp": 410.0,
        "props": {"text": "Unrelated aside", "w": 120, "h": 30},
    },
]
