"""Content recognition placeholder: reads native text only, labels everything else.

No handwriting, equation or diagram recognition exists here. Freehand strokes are reported as
`unrecognized_freehand` placeholders so downstream code never mistakes them for read content.
"""

from typing import Protocol

from edusync_c4.pointing import describe_marking
from edusync_c4.schemas import (
    AnnotationGroup,
    NormalizedAnnotation,
    RecognizedContent,
    ShapeType,
    SlideWord,
    StrokeKind,
)

FREEHAND = {ShapeType.DRAW, ShapeType.HIGHLIGHT}
# Strokes whose kind is recognised even when no slide text is found next to them.
KNOWN_KINDS = {StrokeKind.UNDERLINE, StrokeKind.CIRCLE, StrokeKind.LOOP, StrokeKind.LINE}
# TODO(later phase): VisionOcrRecognizer / math recognizer implementing ContentRecognizer.


class ContentRecognizer(Protocol):
    def recognize(
        self, group: AnnotationGroup, members: list[NormalizedAnnotation]
    ) -> RecognizedContent: ...


class HeuristicRecognizer:
    """Reads native text; recognises underlines and circles and names the slide words they mark.

    `words_by_slide` maps a slide id to its word boxes (whiteboard coordinates). Without words the
    strokes are still classified (an underline, a circle) but point at nothing.
    """

    name = "heuristic-native-text-v2"

    def __init__(self, words_by_slide: dict[str, list[SlideWord]] | None = None) -> None:
        self._words = words_by_slide or {}

    def recognize(
        self, group: AnnotationGroup, members: list[NormalizedAnnotation]
    ) -> RecognizedContent:
        # in the order they were drawn, so sentences read in time order
        freehand = sorted(
            (m for m in members if m.shape_type in FREEHAND),
            key=lambda m: (m.timestamp_seconds, m.shape_id),
        )
        strokes = len(freehand)
        markings = [
            mk
            for m in freehand
            if (mk := describe_marking(m, self._words.get(m.slide_id))) is not None
        ]
        unnamed = strokes - sum(1 for mk in markings if mk.action or mk.kind in KNOWN_KINDS)
        text = group.native_text
        has_other = any(m.shape_type not in FREEHAND for m in members)
        notes: list[str] = []
        if unnamed:
            notes.append(
                f"{unnamed} freehand stroke(s) not recognised: handwriting, equations and "
                "diagrams are not supported yet."
            )
        if not members:
            kind = "empty"
        elif text and strokes:
            kind = "mixed"
        elif text:
            kind = "native_text"
        elif strokes and not has_other:
            kind = "unrecognized_freehand"
        elif strokes:
            kind = "mixed"
        else:
            kind = "shapes_only"
        return RecognizedContent(
            group_id=group.group_id,
            kind=kind,
            text=text,
            freehand_stroke_count=strokes,
            is_placeholder=unnamed > 0 and not text,
            recognizer=self.name,
            notes=notes,
            markings=markings,
        )
