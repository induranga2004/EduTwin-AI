"""BBB HTTP API client (checksummed GET calls)."""

import hashlib
import xml.etree.ElementTree as ET
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx


class BbbApiError(Exception):
    """Raised when the BBB API returns FAILED or an unparsable response."""


class BbbApi(Protocol):
    async def get_meetings(self) -> list[dict[str, str]]: ...


def checksum(call: str, query: str, secret: str, algorithm: str = "sha1") -> str:
    """BBB API checksum: hexdigest of ``call + query + secret``."""
    return hashlib.new(algorithm, (call + query + secret).encode()).hexdigest()


class BbbApiClient:
    """Minimal BBB API client."""

    def __init__(self, base_url: str, secret: str, client: httpx.AsyncClient | None = None) -> None:
        self._base = base_url.rstrip("/")
        self._secret = secret
        self._client = client or httpx.AsyncClient(timeout=10.0)

    def build_url(self, call: str, params: dict[str, Any] | None = None) -> str:
        """Build a signed API URL for ``call``."""
        query = urlencode(params or {})
        sig = checksum(call, query, self._secret)
        full = f"{query}&checksum={sig}" if query else f"checksum={sig}"
        return f"{self._base}/bigbluebutton/api/{call}?{full}"

    async def get_meetings(self) -> list[dict[str, str]]:
        """Return running meetings as flat dicts of child element text."""
        resp = await self._client.get(self.build_url("getMeetings"))
        resp.raise_for_status()
        try:
            root = ET.fromstring(resp.text)
        except ET.ParseError as exc:
            raise BbbApiError(f"Invalid XML from BBB: {exc}") from exc
        if (root.findtext("returncode") or "") != "SUCCESS":
            raise BbbApiError(
                f"BBB call failed: {root.findtext('messageKey')} {root.findtext('message')}"
            )
        return [
            {child.tag: (child.text or "") for child in meeting if len(child) == 0}
            for meeting in root.findall("./meetings/meeting")
        ]
