"""The ASGI entrypoint builds the real app with all four components."""

import pytest
from fastapi.testclient import TestClient

from edusync_api.asgi import app


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize(
    "path", ["/health", "/api/components"] + [f"/api/c{n}/health" for n in range(1, 5)]
)
def test_asgi_app_serves_all_components(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200
