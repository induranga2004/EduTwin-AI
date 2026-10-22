"""Rule-based, template interpretation of an annotation group (explainable baseline).

This is NOT AI understanding. It states what the evidence shows, cites the evidence ids it
used, and says what it could not do. A learned/multimodal interpreter can replace it through
the `ContextualInterpreter` protocol.
"""

from typing import Protocol

from edusync_c4.c2_mock import ComprehensionEvaluator, DifficultyPredictor
from edusync_c4.context import ConceptMatcher, ContextConfig, build_group_context, members_of
from edusync_c4.recognizer import ContentRecognizer, HeuristicRecognizer
from edusync_c4.schemas import (
    AnnotationContext,
    AnnotationGroup,
    AnnotationInterpretation,
    AnnotationTimeline,
    ComprehensionSignal,
    DifficultyEstimate,
    GroupAnalysis,
    RecognizedContent,
    SlideText,
    SpeechSegment,
)
from edusync_core import ConceptRef

METHOD = "template-v1"
TEMPLATE_CONFIDENCE = 0.3  # fixed and deliberately low: a template, not a model
# TODO(later phase): MultimodalInterpreter (slide image + speech + strokes -> explanation).

_CATEGORY = {
    "native_text": "labelled_text",
    "unrecognized_freehand": "freehand_unrecognized",
    "mixed": "text_with_freehand",
    "shapes_only": "shapes_only",
    "empty": "empty",
}


class ContextualInterpreter(Protocol):
    def interpret(
        self, context: AnnotationContext, recognized: RecognizedContent
    ) -> AnnotationInterpretation: ...


def _quote(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class TemplateContextualInterpreter:
    def __init__(
        self,
        difficulty: DifficultyPredictor | None = None,
        comprehension: ComprehensionEvaluator | None = None,
        max_speech_quotes: int = 2,
        max_quote_chars: int = 120,
    ) -> None:
        self._difficulty = difficulty
        self._comprehension = comprehension
        self._max_quotes = max_speech_quotes
        self._max_chars = max_quote_chars

    def interpret(
        self, context: AnnotationContext, recognized: RecognizedContent
    ) -> AnnotationInterpretation:
        ann_id = f"annotation:{context.group_id}"
        cited: list[str] = [ann_id]
        parts = [
            f"On slide {context.slide_id}, an annotation group was drawn between "
            f"{context.span_start:.0f}s and {context.span_end:.0f}s."
        ]

        if recognized.text:
            parts.append(f'Its readable text is "{_quote(recognized.text, self._max_chars)}".')
        if recognized.freehand_stroke_count:
            parts.append(
                f"It contains {recognized.freehand_stroke_count} freehand stroke(s) whose "
                "content is not recognised."
            )

        # Speech: prefer segments overlapping the drawing, then the nearest others.
        def distance(s) -> float:
            if s.relation == "during":
                return 0.0
            return (
                s.start - context.span_end if s.relation == "after" else context.span_start - s.end
            )

        nearest = sorted(context.speech, key=lambda s: (distance(s), s.start))[: self._max_quotes]
        if nearest:
            quotes = "; ".join(
                f'"{_quote(s.text, self._max_chars)}" ({s.relation}, {s.start:.0f}s)'
                for s in sorted(nearest, key=lambda s: s.start)
            )
            parts.append(f"Nearby speech: {quotes}.")
            cited.extend(s.evidence_id for s in nearest)
        else:
            parts.append("No speech was found within the time window.")

        # Concepts: candidates only.
        seen: dict[str, ConceptRef] = {}
        for cand in context.concept_candidates:
            seen.setdefault(cand.concept.id, cand.concept)
            if cand.evidence_id not in cited:
                cited.append(cand.evidence_id)
        if seen:
            labels = ", ".join(c.label for c in seen.values())
            parts.append(
                f"Candidate concept mentions (name found in the text; not confirmed coverage): "
                f"{labels}."
            )

        difficulty: DifficultyEstimate | None = None
        basis = recognized.text or context.slide_text_excerpt
        if self._difficulty is not None and basis:
            difficulty = self._difficulty.predict(basis)
            parts.append(f"Estimated text difficulty {difficulty.score:.2f} ({difficulty.source}).")

        signals: list[ComprehensionSignal] = []
        if self._comprehension is not None:
            for concept_id in seen:
                signal = self._comprehension.evaluate(concept_id)
                if signal is not None:
                    signals.append(signal)
                    state = "struggling" if signal.struggling else "ok"
                    parts.append(
                        f"Quiz signal for {concept_id}: {signal.correct}/{signal.total} "
                        f"correct, {state} ({signal.source})."
                    )

        limitations = ["Rule-based template; no semantic understanding of the annotation."]
        limitations.extend(recognized.notes)

        return AnnotationInterpretation(
            group_id=context.group_id,
            category=_CATEGORY.get(recognized.kind, "shapes_only"),
            summary=" ".join(parts),
            cited_evidence_ids=cited,
            method=METHOD,
            confidence=TEMPLATE_CONFIDENCE,
            difficulty=difficulty,
            comprehension=signals,
            needs_attention=any(s.struggling for s in signals),
            limitations=limitations,
        )


def analyze_groups(
    timeline: AnnotationTimeline,
    groups: list[AnnotationGroup],
    speech: list[SpeechSegment],
    slide_text: SlideText | None,
    concepts: list[ConceptRef],
    config: ContextConfig | None = None,
    matcher: ConceptMatcher | None = None,
    recognizer: ContentRecognizer | None = None,
    interpreter: ContextualInterpreter | None = None,
) -> list[GroupAnalysis]:
    """Evidence -> recognition -> interpretation for each group, in group order."""
    recognizer = recognizer or HeuristicRecognizer()
    interpreter = interpreter or TemplateContextualInterpreter()
    results: list[GroupAnalysis] = []
    for group in groups:
        context = build_group_context(group, speech, slide_text, concepts, config, matcher)
        recognized = recognizer.recognize(group, members_of(group, timeline))
        results.append(
            GroupAnalysis(
                context=context,
                recognized=recognized,
                interpretation=interpreter.interpret(context, recognized),
            )
        )
    return results
