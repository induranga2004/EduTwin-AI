import httpx
import pytest

from edusync_core.bbb.hasura import HasuraError, HttpxHasuraClient, annotations, current_slide_text
from edusync_core.testing import FakeHasuraClient


def _client(handler) -> HttpxHasuraClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return HttpxHasuraClient("http://h/v1/graphql", "sek", http)


async def test_query_posts_and_returns_data():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["secret"] = req.headers["x-hasura-admin-secret"]
        seen["body"] = req.read()
        return httpx.Response(200, json={"data": {"a": 1}})

    assert await _client(handler).query("{a}", {"x": 1}) == {"a": 1}
    assert seen["secret"] == "sek"
    assert b'"query"' in seen["body"]
    assert b'"x"' in seen["body"]


async def test_http_error():
    client = _client(lambda r: httpx.Response(500))
    with pytest.raises(HasuraError):
        await client.query("{a}")


async def test_bad_json():
    client = _client(lambda r: httpx.Response(200, text="nope"))
    with pytest.raises(HasuraError):
        await client.query("{a}")


async def test_graphql_errors():
    client = _client(lambda r: httpx.Response(200, json={"errors": [{"message": "bad"}]}))
    with pytest.raises(HasuraError):
        await client.query("{a}")


async def test_null_data():
    client = _client(lambda r: httpx.Response(200, json={"data": None}))
    assert await client.query("{a}") == {}


async def test_convenience_functions():
    fake = FakeHasuraClient({"pres_page_curr": {"pres_page_curr": []}})
    assert await current_slide_text(fake, "m1") == {"pres_page_curr": []}
    assert await annotations(fake, "m1") == {}
    assert fake.calls[0][1] == {"meetingId": "m1"}
    assert len(fake.calls) == 2


async def test_fake_callable():
    fake = FakeHasuraClient(lambda gql, v: {"v": v})
    assert await fake.query("q", {"a": 1}) == {"v": {"a": 1}}
