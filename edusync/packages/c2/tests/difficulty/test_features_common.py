import pytest

from edusync_c2.difficulty.features._common import assemble
from edusync_c2.difficulty.registry import features_of
from edusync_c2.difficulty.schemas import FeatureGroup, FeatureStatus, FeatureValue

G = FeatureGroup.DENSITY


def _val(name: str, group: FeatureGroup = G) -> FeatureValue:
    return FeatureValue(
        name=name, group=group, value=1.0, status=FeatureStatus.COMPUTED, source="test"
    )


def test_assemble_returns_registry_order() -> None:
    names = features_of(G)
    produced = {n: _val(n) for n in reversed(names)}
    assert [v.name for v in assemble(G, produced)] == list(names)


def test_planned_name_is_filled_as_unavailable() -> None:
    names = features_of(G)
    produced = {n: _val(n) for n in names if n != "dens.visual_complexity"}
    filled = {v.name: v for v in assemble(G, produced)}["dens.visual_complexity"]
    assert filled.name == "dens.visual_complexity"
    assert filled.status is FeatureStatus.UNAVAILABLE
    assert filled.value is None
    assert filled.source == "todo:dens.visual_complexity"
    assert filled.note == "not yet implemented"
    assert filled.group is G


def test_missing_implemented_name_raises() -> None:
    names = features_of(G)
    produced = {n: _val(n) for n in names if n != "dens.words_per_slide"}
    with pytest.raises(ValueError, match=r"density features not produced: dens.words_per_slide"):
        assemble(G, produced)


def test_unknown_extra_name_raises() -> None:
    produced = {n: _val(n) for n in features_of(G)}
    produced["dens.bogus"] = _val("dens.bogus")
    with pytest.raises(ValueError, match=r"not in the registry: dens.bogus"):
        assemble(G, produced)


def test_names_override_and_unregistered_name_is_not_planned() -> None:
    names = ("dens.code_presence", "dens.bogus")
    produced = {n: _val(n) for n in names}
    assert [v.name for v in assemble(G, produced, names)] == list(names)
    with pytest.raises(ValueError, match=r"not produced: dens.bogus"):
        assemble(G, {"dens.code_presence": _val("dens.code_presence")}, names)
