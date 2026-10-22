"""Content recognition placeholder: reads native text only, labels everything else.

No handwriting, equation or diagram recognition exists here. Freehand strokes are reported as
`unrecognized_freehand` placeholders so downstream code never mistakes them for read content.
"""

from typing import Protocol

from edusync_c4.schemas import (
    AnnotationGroup,
    NormalizedAnnotation,
    RecognizedContent,
    ShapeType,
)

FREEHAND = {ShapeType.DRAW, ShapeType.HIGHLIGHT}
# TODO(later phase): VisionOcrRecognizer / math recognizer implementing ContentRecognizer.


class ContentRecognizer(Protocol):
    def recognize(
        self, group: AnnotationGroup, members: list[NormalizedAnnotation]
    ) -> RecognizedContent: ...


class HeuristicRecognizer:
    name = "heuristic-native-text-v1"

    def recognize(
        self, group: AnnotationGroup, members: list[NormalizedAnnotation]
    ) -> RecognizedContent:
        strokes = sum(1 for m in members if m.shape_type in FREEHAND)
        text = group.native_text
        has_other = any(m.shape_type not in FREEHAND for m in members)
        notes: list[str] = []
        if strokes:
            notes.append(
                f"{strokes} freehand stroke(s) not recognised: handwriting, equations and "
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
            is_placeholder=strokes > 0 and not text,
            recognizer=self.name,
            notes=notes,
        )
