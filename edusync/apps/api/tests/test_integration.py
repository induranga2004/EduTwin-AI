"""Integration test against the real c1..c4 components (skipped until they exist)."""

import importlib

import pytest
from fastapi.testclient import TestClient

from edusync_api.main import create_app
from edusync_core import CoreSettings


def components_available() -> bool:
    """True once every edusync_cN package exports COMPONENT."""
    try:
        return all(hasattr(importlib.import_module(f"edusync_c{n}"), "COMPONENT") for n in "1234")
    except ImportError:
        return False


@pytest.mark.skipif(not components_available(), reason="edusync_c1..c4 not exporting COMPONENT")
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_component_health(n: int) -> None:
    client = TestClient(create_app(CoreSettings(_env_file=None)))
    assert client.get(f"/api/c{n}/health").status_code == 200
