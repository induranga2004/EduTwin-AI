"""BBB Hasura GraphQL adapter.

NOTE: the view and column names used below come from bbb-graphql-server/metadata and are
UNVERIFIED against a live BBB instance.
"""

from typing import Any, Protocol

import httpx


class HasuraError(Exception):
    """Raised on HTTP failures or GraphQL ``errors`` from Hasura."""


class HasuraClient(Protocol):
    async def query(self, gql: str, variables: dict[str, Any] | None = None) -> dict[str, Any]: ...


class HttpxHasuraClient:
    """Hasura client POSTing GraphQL over HTTP with the admin secret."""

    def __init__(
        self, url: str, admin_secret: str, client: httpx.AsyncClient | None = None
    ) -> None:
        self._url = url
        self._secret = admin_secret
        self._client = client or httpx.AsyncClient(timeout=10.0)

    async def query(self, gql: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = {"query": gql, "variables": variables or {}}
        headers = {"x-hasura-admin-secret": self._secret}
        try:
            resp = await self._client.post(self._url, json=payload, headers=headers)
            resp.raise_for_status()
            body = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HasuraError(f"Hasura request failed: {exc}") from exc
        if body.get("errors"):
            raise HasuraError(f"GraphQL errors: {body['errors']}")
        return body.get("data") or {}


CURRENT_SLIDE_QUERY = """
query CurrentSlide($meetingId: String!) {
  pres_page_curr(where: {meetingId: {_eq: $meetingId}}) {
    content num presentationId pageId urlsJson
  }
}
"""

ANNOTATIONS_QUERY = """
query Annotations($meetingId: String!) {
  pres_annotation_history_curr(where: {meetingId: {_eq: $meetingId}}) {
    annotationId pageId userId annotationInfo updatedAt
  }
}
"""


async def current_slide_text(client: HasuraClient, meeting_id: str) -> dict[str, Any]:
    """Fetch the current slide page(s) of a meeting."""
    return await client.query(CURRENT_SLIDE_QUERY, {"meetingId": meeting_id})


async def annotations(client: HasuraClient, meeting_id: str) -> dict[str, Any]:
    """Fetch whiteboard annotations of a meeting."""
    return await client.query(ANNOTATIONS_QUERY, {"meetingId": meeting_id})
