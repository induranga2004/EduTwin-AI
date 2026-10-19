"""Test doubles for the BBB adapters, imported by other packages' tests."""

from collections.abc import Callable
from typing import Any

from edusync_core.bbb.events import InMemoryEventBus

__all__ = ["FakeBbbApi", "FakeHasuraClient", "InMemoryEventBus"]


class FakeHasuraClient:
    """Returns canned responses and records every call.

    ``responses`` maps a substring of the GraphQL text to a data dict, or is a callable
    ``(gql, variables) -> dict``.
    """

    def __init__(
        self,
        responses: dict[str, dict[str, Any]] | Callable[[str, dict[str, Any] | None], dict],
    ) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def query(self, gql: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        self.calls.append((gql, variables))
        if callable(self.responses):
            return self.responses(gql, variables)
        for key, data in self.responses.items():
            if key in gql:
                return data
        return {}


class FakeBbbApi:
    """Serves a fixed list of meetings."""

    def __init__(self, meetings: list[dict[str, str]] | None = None) -> None:
        self.meetings = meetings or []

    async def get_meetings(self) -> list[dict[str, str]]:
        return list(self.meetings)
