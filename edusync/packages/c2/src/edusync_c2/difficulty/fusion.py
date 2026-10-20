"""Fusion input assembly: ordered features with missing masks plus the concept text."""

from edusync_c2.difficulty.registry import FEATURE_SPECS
from edusync_c2.difficulty.schemas import (
    ConceptRecord,
    FeatureGroup,
    FeatureStatus,
    FeatureValue,
    FusionFeature,
    FusionInput,
    GroupSummary,
    SlideRecord,
)


def build_fusion_input(
    concept: ConceptRecord,
    features: list[FeatureValue],
    concept_slides: list[SlideRecord],
    *,
    use_mocked: bool = False,
    max_text_chars: int = 1500,
) -> FusionInput:
    """Order features by registry; unavailable (and by default mocked) ones are missing."""
    by_name = {f.name: f for f in features}
    out: list[FusionFeature] = []
    for spec in FEATURE_SPECS:
        f = by_name.get(spec.name)
        if f is None:
            out.append(
                FusionFeature(
                    name=spec.name, value=None, status=FeatureStatus.UNAVAILABLE, missing=True
                )
            )
            continue
        missing = f.value is None or f.status is FeatureStatus.UNAVAILABLE
        missing = missing or (f.status is FeatureStatus.MOCKED and not use_mocked)
        out.append(
            FusionFeature(
                name=f.name, value=None if missing else f.value, status=f.status, missing=missing
            )
        )
    head = f"{concept.label}. {concept.description or ''}\n"
    body = "\n".join(s.text for s in concept_slides)
    text = head + body[: max(0, max_text_chars - len(head))]
    return FusionInput(concept_id=concept.id, text=text, features=out)


def to_vector(fusion: FusionInput) -> tuple[list[float], list[int]]:
    """Values (0.0 for missing) and mask (1 = missing) in registry order."""
    values = [0.0 if f.missing or f.value is None else f.value for f in fusion.features]
    mask = [1 if f.missing else 0 for f in fusion.features]
    return values, mask


def summarize_features(features: list[FeatureValue]) -> dict[FeatureGroup, GroupSummary]:
    """Counts per group and status; all four groups always present."""
    out = {g: GroupSummary() for g in FeatureGroup}
    for f in features:
        s = out[f.group]
        if f.status is FeatureStatus.COMPUTED:
            s.computed += 1
        elif f.status is FeatureStatus.MOCKED:
            s.mocked += 1
        else:
            s.unavailable += 1
    return out
