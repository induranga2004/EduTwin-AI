import asyncio
import json

import pytest

from edusync_core.bbb.events import FROM_AKKA_APPS, TO_AKKA_APPS, InMemoryEventBus, parse_envelope

ENV = {
    "envelope": {"name": "PresentationPageChangedEvtMsg", "routing": {}},
    "core": {"header": {}, "body": {"pageId": "p1"}},
}
BAD_TYPES = {"envelope": {"name": 1}, "core": {"body": {}}}


def test_parse_ok_str_and_bytes():
    ev = parse_envelope("ch", json.dumps(ENV))
    assert ev.name == "PresentationPageChangedEvtMsg"
    assert ev.body == {"pageId": "p1"}
    assert parse_envelope("ch", json.dumps(ENV).encode()).channel == "ch"


@pytest.mark.parametrize(
    "raw", ["not json", b"\xff", "[]", '{"envelope": {}}', json.dumps(BAD_TYPES)]
)
def test_parse_bad(raw):
    with pytest.raises(ValueError, match="Malformed"):
        parse_envelope("ch", raw)


async def test_in_memory_delivers_to_channel_subscribers():
    bus = InMemoryEventBus()
    it = bus.subscribe([FROM_AKKA_APPS])
    task = asyncio.ensure_future(it.__anext__())
    await asyncio.sleep(0)
    await bus.publish(TO_AKKA_APPS, ENV)  # other channel: ignored
    await bus.publish(FROM_AKKA_APPS, ENV)
    ev = await asyncio.wait_for(task, 1)
    assert ev.name == "PresentationPageChangedEvtMsg"
    await it.aclose()
    assert bus._queues[FROM_AKKA_APPS] == []


async def test_in_memory_plain_message():
    bus = InMemoryEventBus()
    it = bus.subscribe(["c"])
    task = asyncio.ensure_future(it.__anext__())
    await asyncio.sleep(0)
    await bus.publish("c", {"name": "x", "body": {"k": 1}})
    ev = await asyncio.wait_for(task, 1)
    assert (ev.name, ev.body) == ("x", {"k": 1})
    await it.aclose()
