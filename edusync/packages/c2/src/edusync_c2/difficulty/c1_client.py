"""HTTP transport to C1's graph route (stdlib only, no C1 code imported).

``UrllibTransport`` posts the uploaded file as multipart/form-data (field ``file``) to
``{base_url}/lectures/graph`` and returns the parsed JSON object. Failures become
``GraphProviderError``. File content and file names are never logged or put in messages.
"""

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable

from edusync_c2.difficulty.errors import GraphProviderError
from edusync_c2.logger import get_logger

log = get_logger("c1_client")

Transport = Callable[[bytes, str], dict]

_DETAIL_MAX = 200
_CONTENT_TYPES = {
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
_UNSAFE = re.compile('["\\r\\n\\\\]')


def _safe_filename(filename: str) -> str:
    base = re.split("[\\\\/]", filename)[-1]
    return _UNSAFE.sub("", base) or "upload"


def _content_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _CONTENT_TYPES.get(ext, "application/octet-stream")


def build_multipart(data: bytes, filename: str) -> tuple[bytes, str]:
    """Return ``(body, content_type_header)``; the boundary is derived from the payload."""
    boundary = "edusync-c2-" + hashlib.sha256(data).hexdigest()[:32]
    name = _safe_filename(filename)
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'
        f"Content-Type: {_content_type(name)}\r\n\r\n"
    ).encode()
    body = head + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _fail(status: int, message: str, reason: str) -> GraphProviderError:
    log.warning("c1_request_failed status=%d reason=%s", status, reason)
    return GraphProviderError(status, message)


def _client_error_detail(err: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(err.read()).get("detail")
    except (ValueError, AttributeError, OSError):
        detail = None
    if isinstance(detail, str) and detail.strip():
        return detail.strip()[:_DETAIL_MAX]
    return "C1 rejected the file"


class UrllibTransport:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self._url = base_url.rstrip("/") + "/lectures/graph"
        self._timeout_s = timeout_s

    def __call__(self, data: bytes, filename: str) -> dict:
        body, content_type = build_multipart(data, filename)
        request = urllib.request.Request(
            self._url, data=body, method="POST", headers={"Content-Type": content_type}
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as err:
            if 400 <= err.code < 500:
                raise _fail(422, _client_error_detail(err), f"c1_http_{err.code}") from None
            raise _fail(502, "C1 failed to build the graph", f"c1_http_{err.code}") from None
        except TimeoutError:
            raise _fail(504, "C1 timed out", "timeout") from None
        except urllib.error.URLError as err:
            if isinstance(err.reason, TimeoutError):
                raise _fail(504, "C1 timed out", "timeout") from None
            raise _fail(502, "C1 is unreachable", "unreachable") from None
        except OSError:
            raise _fail(502, "C1 is unreachable", "connection_error") from None
        try:
            parsed = json.loads(raw)
        except ValueError:
            raise _fail(502, "C1 returned an invalid response", "invalid_json") from None
        if not isinstance(parsed, dict):
            raise _fail(502, "C1 returned an invalid response", "not_an_object")
        return parsed
