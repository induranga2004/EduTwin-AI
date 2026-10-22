"""Plain-language reasons, slide scores and the ranked hotspot list. No slide text is used."""

from collections.abc import Mapping, Sequence

from edusync_c2.difficulty.scoring.combiner import label_for
from edusync_c2.difficulty.scoring.signals import FeatureScore

REASON_MIN_EFFECT = 0.6  # a signal must beat 60% of the reference to be named as a reason
MAX_REASONS = 3
CONCEPT_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
TEXT_ONLY_CONFIDENCE = 0.3
NO_SLIDE = 10**9
NO_REASON = "No single signal stands out"
TEXT_REASON = "Wording reads as difficult to the text models"

PHRASES = {
    "lang.unfamiliar_vocab_ratio": "share of unfamiliar words",
    "lang.subword_fragmentation": "word fragmentation (rare vocabulary)",
    "lang.domain_term_ratio": "share of technical terms",
    "lang.lexical_score": "lexical difficulty",
    "prereq.chain_depth": "depth of the prerequisite chain",
    "prereq.direct_count": "number of direct prerequisites",
    "prereq.transitive_count": "number of prerequisites in total",
    "prereq.dependents_count": "number of concepts building on it",
    "expl.definition_count": "number of definitions",
    "expl.example_count": "number of worked examples",
    "expl.step_count": "number of step-by-step markers",
    "expl.diagram_support": "pictures and charts supporting it",
    "dens.words_per_slide": "words per slide",
    "dens.min_font_pt": "smallest font size",
    "dens.visual_complexity": "visual complexity",
}


def phrase(name: str) -> str:
    return PHRASES.get(name) or name.split(".", 1)[1].replace("_", " ")


def concept_reasons(features: Sequence[FeatureScore], text_percentile: float | None) -> list[str]:
    """Up to three strongest difficulty cues, strongest first (ties by name)."""
    candidates: list[tuple[float, str, str]] = []
    for f in features:
        if f.effect is not None and f.effect >= REASON_MIN_EFFECT:
            level = "high" if f.direction > 0 else "low"
            text = f"{phrase(f.name)} is {level}"
            candidates.append(
                (
                    f.effect,
                    f.name,
                    f"{text} (harder than {round(f.effect * 100)}% of reference concepts)",
                )
            )
    if text_percentile is not None and text_percentile >= REASON_MIN_EFFECT:
        candidates.append(
            (
                text_percentile,
                "~text",
                f"{TEXT_REASON} ({round(text_percentile * 100)}th percentile)",
            )
        )
    candidates.sort(key=lambda c: (-c[0], c[1]))
    return [c[2] for c in candidates[:MAX_REASONS]] or [NO_REASON]


def slide_score(concept_scores: Sequence[float], text_percentile: float | None) -> float | None:
    """0.6 * max concept score + 0.4 * text percentile; whichever exists if only one does."""
    if concept_scores and text_percentile is not None:
        return CONCEPT_WEIGHT * max(concept_scores) + TEXT_WEIGHT * text_percentile
    if concept_scores:
        return max(concept_scores)
    return text_percentile


def build_slide_scores(
    slide_indexes: Sequence[int],
    concepts_on_slide: Mapping[int, Sequence[str]],
    concepts: Mapping[str, dict],
    text_percentiles: Mapping[int, float | None],
    low: float,
    high: float,
) -> list[dict]:
    """One row per slide: score, label, confidence, concept ids and reasons."""
    rows = []
    for index in slide_indexes:
        ids = sorted(
            concepts_on_slide.get(index, ()),
            key=lambda i: (-(concepts[i]["score"] or 0.0), i),
        )
        known = [i for i in ids if concepts[i]["score"] is not None]
        text_pct = text_percentiles.get(index)
        score = slide_score([concepts[i]["score"] for i in known], text_pct)
        confidence = (
            sum(concepts[i]["confidence"] for i in ids) / len(ids) if ids else TEXT_ONLY_CONFIDENCE
        )
        reasons = []
        if known:
            top = concepts[known[0]]
            reasons.append(
                f"Contains the concept '{top['concept_label']}' (score {round(top['score'] * 100)})"
            )
        if text_pct is not None and text_pct >= REASON_MIN_EFFECT:
            reasons.append(f"{TEXT_REASON} ({round(text_pct * 100)}th percentile)")
        rows.append(
            {
                "index": index,
                "score": score,
                "label": label_for(score, low, high),
                "confidence": 0.0 if score is None else confidence,
                "concept_ids": ids,
                "reasons": reasons or [NO_REASON],
            }
        )
    return rows


def build_hotspots(
    concept_rows: Sequence[dict], slide_rows: Sequence[dict], top_n: int
) -> list[dict]:
    """Top ``top_n`` concepts and top ``top_n`` slides, merged and ranked by score.

    Ties: concepts before slides, then the smallest first slide / slide index, then id.
    """
    first = lambda c: min(c["slide_indexes"], default=NO_SLIDE)  # noqa: E731
    concepts = sorted(
        (c for c in concept_rows if c["score"] is not None),
        key=lambda c: (-c["score"], first(c), c["concept_id"]),
    )[:top_n]
    slides = sorted(
        (s for s in slide_rows if s["score"] is not None), key=lambda s: (-s["score"], s["index"])
    )[:top_n]
    keyed = [
        (
            (-c["score"], 0, first(c), c["concept_id"]),
            {
                "kind": "concept",
                "concept_id": c["concept_id"],
                "concept_label": c["concept_label"],
                "slides": list(c["slide_indexes"]),
                "score": c["score"],
                "label": c["label"],
                "confidence": c["confidence"],
                "reasons": c["reasons"],
            },
        )
        for c in concepts
    ] + [
        (
            (-s["score"], 1, s["index"], ""),
            {
                "kind": "slide",
                "slide_index": s["index"],
                "slides": [s["index"]],
                "score": s["score"],
                "label": s["label"],
                "confidence": s["confidence"],
                "reasons": s["reasons"],
            },
        )
        for s in slides
    ]
    keyed.sort(key=lambda pair: pair[0])
    return [{"rank": rank, **item} for rank, (_, item) in enumerate(keyed, start=1)]
