"""Shared assembly of a feature group's values, always by registry name (never by position)."""

from collections.abc import Iterable

from edusync_c2.difficulty.registry import features_of, spec_of
from edusync_c2.difficulty.schemas import FeatureGroup, FeatureStatus, FeatureValue


def assemble(
    group: FeatureGroup,
    produced: dict[str, FeatureValue],
    names: Iterable[str] | None = None,
) -> list[FeatureValue]:
    """Return the group's values in registry order.

    ``produced`` maps feature name -> value. A registry name that is missing from ``produced`` and
    is planned (not built yet) becomes an unavailable value with source ``todo:<name>``. A missing
    implemented or mock name, or a produced name outside the group, raises ``ValueError``.
    ``names`` overrides the registry order (used by tests that extend a group).
    """
    ordered = tuple(features_of(group) if names is None else names)
    extra = [n for n in produced if n not in ordered]
    if extra:
        raise ValueError(f"{group.value} features not in the registry: {', '.join(extra)}")
    out: list[FeatureValue] = []
    missing: list[str] = []
    for name in ordered:
        if name in produced:
            out.append(produced[name])
        elif _is_planned(name):
            out.append(
                FeatureValue(
                    name=name,
                    group=group,
                    value=None,
                    status=FeatureStatus.UNAVAILABLE,
                    source=f"todo:{name}",
                    note="not yet implemented",
                )
            )
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"{group.value} features not produced: {', '.join(missing)}")
    return out


def _is_planned(name: str) -> bool:
    try:
        return spec_of(name).maturity == "planned"
    except KeyError:
        return False
