import pytest

from edusync_c2.difficulty.features import _common as common
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


def test_planned_name_is_filled_as_unavailable(monkeypatch) -> None:
    # Every registered feature is implemented now, so mark one as planned for this test only.
    target = "expl.semantic_definition_count"
    real_spec_of = common.spec_of
    monkeypatch.setattr(
        common,
        "spec_of",
        lambda name: (
            real_spec_of(name).model_copy(update={"maturity": "planned"})
            if name == target
            else real_spec_of(name)
        ),
    )
    group = FeatureGroup.EXPLANATION
    names = features_of(group)
    assert target not in common.built_features_of(group)
    produced = {n: _val(n, group) for n in names if n != target}
    filled = {v.name: v for v in assemble(group, produced)}[target]
    assert filled.name == target
    assert filled.status is FeatureStatus.UNAVAILABLE
    assert filled.value is None
    assert filled.source == f"todo:{target}"
    assert filled.note == "not yet implemented"
    assert filled.group is group


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
