"""Evidence merging and speech/visual reconciliation (rule-based baseline)."""

from edusync_core import EvidenceItem, EvidenceSource, Flag, FlagSeverity

VISUAL_SOURCES = frozenset({EvidenceSource.SLIDE, EvidenceSource.WHITEBOARD})


def merge_evidence(items: list[EvidenceItem]) -> dict[str, list[EvidenceItem]]:
    """Group by concept id, sort by timestamp, drop exact (source, seconds, text) duplicates."""
    grouped: dict[str, list[EvidenceItem]] = {}
    seen: set[tuple[str, EvidenceSource, float, str]] = set()
    for item in sorted(items, key=lambda i: i.timestamp.seconds):
        key = (item.concept.id, item.source, item.timestamp.seconds, item.text)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(item.concept.id, []).append(item)
    return grouped


def reconcile(
    speech: list[EvidenceItem],
    visual: list[EvidenceItem],
    window_seconds: float = 120.0,
) -> list[Flag]:
    """Flag concepts shown visually (slide/whiteboard) with no speech within the window.

    One REVIEW flag per concept, anchored at its earliest unmatched visual item.
    """
    speech_by_concept = merge_evidence(speech)
    flags: list[Flag] = []
    flagged: set[str] = set()
    for group in merge_evidence(visual).values():
        for v in group:
            cid = v.concept.id
            if v.source not in VISUAL_SOURCES or cid in flagged:
                continue
            near = [
                s
                for s in speech_by_concept.get(cid, [])
                if abs(s.timestamp.seconds - v.timestamp.seconds) <= window_seconds
            ]
            if near:
                continue
            flagged.add(cid)
            flags.append(
                Flag(
                    concept=v.concept,
                    timestamp=v.timestamp,
                    component="c4",
                    severity=FlagSeverity.REVIEW,
                    reason=(
                        f"Shown on {v.source.value} but not mentioned in speech "
                        f"within {window_seconds:.0f}s."
                    ),
                    evidence=[v],
                )
            )
    return sorted(flags, key=lambda f: f.timestamp.seconds)
