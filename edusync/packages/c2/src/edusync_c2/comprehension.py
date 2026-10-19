"""Comprehension validation from poll/quiz tallies (rule-based baseline)."""

from edusync_core import ConceptRef, Flag, FlagSeverity, LectureTimestamp

DIFFICULTY_BOOST = 0.1


def effective_threshold(threshold: float, predicted_difficulty: float) -> float:
    """Raise the threshold by up to DIFFICULTY_BOOST for hard concepts (capped at 1.0)."""
    difficulty = max(0.0, min(1.0, predicted_difficulty))
    return min(1.0, threshold + DIFFICULTY_BOOST * difficulty)


def flag_for_clarification(
    concept: ConceptRef,
    correct: int,
    total: int,
    timestamp: LectureTimestamp,
    predicted_difficulty: float = 0.0,
    threshold: float = 0.6,
    min_responses: int = 3,
) -> Flag | None:
    """Flag a concept when the correct ratio is below the effective threshold.

    Rule: effective = threshold + 0.1 * predicted_difficulty (capped at 1). Ratio below
    effective -> REVIEW; ratio below effective / 2 -> URGENT. Fewer than ``min_responses``
    responses -> None (not enough evidence).
    """
    if total < min_responses or total <= 0:
        return None
    ratio = correct / total
    limit = effective_threshold(threshold, predicted_difficulty)
    if ratio >= limit:
        return None
    severity = FlagSeverity.URGENT if ratio < limit / 2 else FlagSeverity.REVIEW
    return Flag(
        concept=concept,
        timestamp=timestamp,
        component="c2",
        severity=severity,
        reason=(
            f"Only {correct}/{total} correct ({ratio:.0%}), below {limit:.0%} "
            f"(predicted difficulty {predicted_difficulty:.2f}); consider clarifying."
        ),
    )
