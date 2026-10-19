import hashlib

import httpx
import pytest

from edusync_core.bbb.api import BbbApiClient, BbbApiError, checksum
from edusync_core.testing import FakeBbbApi


def test_checksum_known_answer():
    # sha1("getMeetingsabc"), precomputed
    assert checksum("getMeetings", "", "abc") == hashlib.sha1(b"getMeetingsabc").hexdigest()
    assert len(checksum("x", "a=1", "s", "sha256")) == 64


def test_build_url():
    api = BbbApiClient("https://bbb.example/", "s3")
    url = api.build_url("create", {"name": "A B", "meetingID": "m"})
    q = "name=A+B&meetingID=m"
    sig = checksum("create", q, "s3")
    assert url == f"https://bbb.example/bigbluebutton/api/create?{q}&checksum={sig}"
    assert api.build_url("getMeetings").endswith(f"?checksum={checksum('getMeetings', '', 's3')}")


def _api(xml: str, status: int = 200) -> BbbApiClient:
    transport = httpx.MockTransport(lambda r: httpx.Response(status, text=xml))
    return BbbApiClient("http://bbb", "s", httpx.AsyncClient(transport=transport))


async def test_get_meetings():
    xml = (
        "<response><returncode>SUCCESS</returncode><meetings>"
        "<meeting><meetingID>m1</meetingID><meetingName>Intro</meetingName>"
        "<attendees><attendee><userID>u</userID></attendee></attendees></meeting>"
        "</meetings></response>"
    )
    assert await _api(xml).get_meetings() == [{"meetingID": "m1", "meetingName": "Intro"}]


async def test_get_meetings_failed():
    xml = (
        "<response><returncode>FAILED</returncode><messageKey>checksumError</messageKey></response>"
    )
    with pytest.raises(BbbApiError):
        await _api(xml).get_meetings()


async def test_get_meetings_bad_xml_and_http():
    with pytest.raises(BbbApiError):
        await _api("<nope").get_meetings()
    with pytest.raises(httpx.HTTPStatusError):
        await _api("", 500).get_meetings()


async def test_fake_api():
    assert await FakeBbbApi([{"meetingID": "m"}]).get_meetings() == [{"meetingID": "m"}]
    assert await FakeBbbApi().get_meetings() == []
