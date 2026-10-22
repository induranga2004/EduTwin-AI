"""Local "What it means" summaries: rule-based, built only from real evidence.

No language model is involved and nothing leaves the computer. For one group of annotations it
states what was written or drawn, writes a diagram as a sentence, and quotes the lecturer's most
relevant spoken sentences and the matching line of slide text. "Relevant" means sharing words
with what was written (simple word overlap), falling back to what was said at that moment.

Everything shown is a quote or a plain statement of what the data contains; nothing is invented.
Because it is only word overlap, it can pick a sentence that shares a word but is off-topic, and it
cannot explain *why* something was drawn. A generative interpreter can replace it through the
`ContextualInterpreter` protocol and its output is stored with the same models.
"""

import re

from edusync_c4.interpreter import ContextualInterpreter, analyze_groups
from edusync_c4.recognizer import HeuristicRecognizer
from edusync_c4.schemas import (
    AnnotationContext,
    AnnotationGroup,
    AnnotationInterpretation,
    AnnotationTimeline,
    GroupAnalysis,
    RecognizedContent,
    SlideText,
    SpeechEvidence,
    SpeechSegment,
)

METHOD = "evidence-local-v1"
MAX_QUOTES = 2
MAX_QUOTE_CHARS = 160
MAX_SLIDE_LINE_CHARS = 200
CONFIDENCE = 0.3  # fixed: this is word overlap, not understanding

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
STOP_WORDS = frozenset(
    """
    the and for are but not you all any can had her was one our out has his how its may new now
    see two who did let say she too use that this with from have they will what when where
    which their there then than them these those would could should about after also been into
    just like more most only over such some very your because while being both each other
    """.split()  # noqa: SIM905 - kept as readable prose
)


def _clock(seconds: float) -> str:
    s = max(0, round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _stem(word: str) -> str:
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    for suffix in ("ing", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def terms(text: str) -> set[str]:
    """Lower-cased, lightly stemmed content words of `text` (no stop words)."""
    return {
        _stem(w.lower().strip("'-"))
        for w in _WORD.findall(text or "")
        if w.lower() not in STOP_WORDS
    }


def _trim(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def rank_speech(
    context: AnnotationContext, target: set[str], limit: int = MAX_QUOTES
) -> list[SpeechEvidence]:
    """The most relevant spoken segments: shared words first, then closeness in time."""

    def distance(s: SpeechEvidence) -> float:
        if s.relation == "during":
            return 0.0
        if s.relation == "after":
            return s.start - context.span_end
        return context.span_start - s.end

    usable = [s for s in context.speech if s.text.strip()]
    scored = [(len(target & terms(s.text)), s) for s in usable]
    related = sorted(
        (item for item in scored if item[0] > 0),
        key=lambda item: (-item[0], distance(item[1]), item[1].start),
    )
    if related:
        # keep only sentences about as relevant as the best one, not weak one-word matches
        floor = max(1, (related[0][0] + 1) // 2)
        chosen = [s for score, s in related if score >= floor][:limit]
    else:
        chosen = sorted(usable, key=lambda s: (distance(s), s.start))[:limit]
    return sorted(chosen, key=lambda s: s.start)


def pick_relevant(segments, target: set[str], limit: int):
    """From time-nearby segments, the ones sharing the most words with `target`.

    Falls back to the earliest ones when nothing shares a word. Works on any object with `.text`
    and `.start`.
    """
    scored = [(len(target & terms(s.text)), s) for s in segments]
    related = sorted((item for item in scored if item[0] > 0), key=lambda i: (-i[0], i[1].start))
    if related:
        floor = max(1, (related[0][0] + 1) // 2)
        chosen = [s for score, s in related if score >= floor][:limit]
    else:
        chosen = sorted(segments, key=lambda s: s.start)[:limit]
    return sorted(chosen, key=lambda s: s.start)


def best_slide_line(slide_text: str | None, target: set[str]) -> str | None:
    """The slide line sharing the most words with `target`, or None if none shares any."""
    if not slide_text or not target:
        return None
    best: tuple[int, int, str] | None = None
    for raw in _SENTENCE_SPLIT.split(slide_text):
        line = raw.strip().lstrip("▪•-* ").strip()
        if len(line) < 12:
            continue
        overlap = len(target & terms(line))
        if overlap and (best is None or (overlap, -len(line)) > (best[0], -best[1])):
            best = (overlap, len(line), line)
    return _trim(best[2], MAX_SLIDE_LINE_CHARS) if best else None


def _lead(context: AnnotationContext, recognized: RecognizedContent) -> str:
    when = _clock(context.span_start)
    text = (recognized.text or "").strip()
    strokes = recognized.freehand_stroke_count
    plural = "s" if strokes != 1 else ""
    if context.diagram_sentences:
        listed = "; ".join(context.diagram_sentences)
        return f"At {when} the lecturer drew a diagram: {listed}."
    if recognized.kind == "native_text":
        return f"At {when} the lecturer wrote “{_trim(text, 200)}”."
    if recognized.kind == "mixed":
        written = f"wrote “{_trim(text, 200)}” and " if text else ""
        return (
            f"At {when} the lecturer {written}drew {strokes} freehand stroke{plural} "
            "(the drawing itself is not recognised)."
        )
    if recognized.kind == "unrecognized_freehand":
        return (
            f"At {when} the lecturer drew {strokes} freehand stroke{plural} "
            "(the drawing itself is not recognised)."
        )
    if recognized.kind == "shapes_only":
        return f"At {when} the lecturer drew shapes with no text."
    return f"At {when} the lecturer made an annotation that is no longer on the board."


class EvidenceInterpreter:
    """Rule-based explanation from evidence only. Implements `ContextualInterpreter`."""

    def interpret(
        self, context: AnnotationContext, recognized: RecognizedContent
    ) -> AnnotationInterpretation:
        ann_id = f"annotation:{context.group_id}"
        cited = [ann_id]
        cited.extend(f"diagram:{i}" for i in range(len(context.diagram_sentences)))
        target = terms(recognized.text or "") | terms(" ".join(context.diagram_sentences))

        parts = [_lead(context, recognized)]
        quotes = rank_speech(context, target)
        if quotes:
            said = "; ".join(
                f"“{_trim(q.text, MAX_QUOTE_CHARS)}” ({_clock(q.start)})" for q in quotes
            )
            parts.append(f"Around that moment the lecturer said: {said}.")
            cited.extend(q.evidence_id for q in quotes)
        else:
            parts.append("No speech was found near this moment.")

        speech_terms = set().union(*(terms(q.text) for q in quotes)) if quotes else set()
        line = best_slide_line(context.slide_text_excerpt, target | speech_terms)
        if line and context.slide_evidence_id:
            parts.append(f"Related slide text: “{line}”.")
            cited.append(context.slide_evidence_id)

        limitations = [
            "Rule-based summary, not AI: quotes are chosen by shared words and closeness in time, "
            "so a quote can be off-topic and the reason for the drawing is not explained.",
            *recognized.notes,
        ]
        category = "diagram" if context.diagram_sentences else recognized.kind
        return AnnotationInterpretation(
            group_id=context.group_id,
            category=category,
            summary=" ".join(parts),
            cited_evidence_ids=cited,
            method=METHOD,
            confidence=CONFIDENCE,
            limitations=limitations,
        )


def explain_slide(
    timeline: AnnotationTimeline,
    groups: list[AnnotationGroup],
    speech: list[SpeechSegment] | None,
    slide_text: str | None,
    interpreter: ContextualInterpreter | None = None,
) -> list[GroupAnalysis]:
    """Explain every group of one slide from real inputs only (no mock data, no concepts)."""
    from edusync_c4.context import ContextConfig

    slide = SlideText(slide_id=timeline.slide_id, text=slide_text) if slide_text else None
    return analyze_groups(
        timeline,
        groups,
        speech or [],
        slide,
        [],
        config=ContextConfig(max_slide_excerpt_chars=5000),
        recognizer=HeuristicRecognizer(),
        interpreter=interpreter or EvidenceInterpreter(),
    )
