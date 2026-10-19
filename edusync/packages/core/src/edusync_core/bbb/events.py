"""BBB Redis pub/sub events: envelope parsing and event bus implementations."""

import asyncio
import json
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any, Protocol

from pydantic import BaseModel, Field

FROM_AKKA_APPS = "from-akka-apps-redis-channel"
FROM_AKKA_FRONTEND = "from-akka-apps-frontend-redis-channel"
FROM_AKKA_PRES = "from-akka-apps-pres-redis-channel"
TO_AKKA_APPS = "to-akka-apps-redis-channel"


class BbbEvent(BaseModel):
    """A parsed BBB message: channel, event name and body."""

    channel: str
    name: str
    body: dict[str, Any] = Field(default_factory=dict)


def parse_envelope(channel: str, raw: str | bytes) -> BbbEvent:
    """Parse a BBB Redis JSON envelope; raise ValueError on bad JSON or shape."""
    try:
        data = json.loads(raw)
        name = data["envelope"]["name"]
        body = data["core"]["body"]
    except (ValueError, KeyError, TypeError) as exc:  # JSONDecodeError is a ValueError
        raise ValueError(f"Malformed BBB envelope: {exc!r}") from exc
    if not isinstance(name, str) or not isinstance(body, dict):
        raise ValueError("Malformed BBB envelope: bad name/body types")
    return BbbEvent(channel=channel, name=name, body=body)


class EventBus(Protocol):
    async def publish(self, channel: str, message: dict[str, Any]) -> None: ...

    def subscribe(self, channels: list[str]) -> AsyncIterator[BbbEvent]: ...


class RedisEventBus:  # pragma: no cover
    """Event bus backed by Redis pub/sub."""

    def __init__(self, url: str) -> None:
        import redis.asyncio as aioredis

        self._redis = aioredis.from_url(url)

    async def publish(self, channel: str, message: dict[str, Any]) -> None:
        await self._redis.publish(channel, json.dumps(message))

    async def subscribe(self, channels: list[str]) -> AsyncIterator[BbbEvent]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(*channels)
        try:
            async for msg in pubsub.listen():
                if msg["type"] != "message":
                    continue
                chan = msg["channel"]
                chan = chan.decode() if isinstance(chan, bytes) else chan
                try:
                    yield parse_envelope(chan, msg["data"])
                except ValueError:
                    continue
        finally:
            await pubsub.aclose()


class InMemoryEventBus:
    """Queue-based bus for tests: ``publish`` delivers to subscribers of that channel.

    Messages are BBB envelopes; plain dicts with ``name``/``body`` keys are accepted too.
    """

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue[BbbEvent]]] = defaultdict(list)

    async def publish(self, channel: str, message: dict[str, Any]) -> None:
        try:
            event = parse_envelope(channel, json.dumps(message))
        except ValueError:
            event = BbbEvent(
                channel=channel, name=str(message.get("name", "")), body=message.get("body", {})
            )
        for queue in list(self._queues[channel]):
            queue.put_nowait(event)

    async def subscribe(self, channels: list[str]) -> AsyncIterator[BbbEvent]:
        queue: asyncio.Queue[BbbEvent] = asyncio.Queue()
        for chan in channels:
            self._queues[chan].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            for chan in channels:
                self._queues[chan].remove(queue)
