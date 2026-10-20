import pytest
from pydantic import ValidationError

from edusync_c2.difficulty.schemas import FeatureGroup, FeatureStatus, FeatureValue


def make(status, value):
    return FeatureValue(
        name="x", group=FeatureGroup.LANGUAGE, value=value, status=status, source="t"
    )


def test_unavailable_with_value_rejected():
    with pytest.raises(ValidationError, match="unavailable"):
        make(FeatureStatus.UNAVAILABLE, 1.0)


def test_computed_without_value_rejected():
    with pytest.raises(ValidationError, match="computed"):
        make(FeatureStatus.COMPUTED, None)


def test_valid_combinations():
    assert make(FeatureStatus.UNAVAILABLE, None).value is None
    assert make(FeatureStatus.COMPUTED, 0.0).value == 0.0
    assert make(FeatureStatus.MOCKED, None).value is None
    assert make(FeatureStatus.MOCKED, 0.4).value == 0.4
