"""C1 transport: multipart body and error mapping against a localhost stdlib server."""

import hashlib
import json
import logging
import socket
import threading
import time
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest

from edusync_c2.difficulty import c1_client
from edusync_c2.difficulty.c1_client import UrllibTransport, build_multipart
from edusync_c2.difficulty.errors import GraphProviderError

PPTX_CT = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def test_multipart_exact_bytes() -> None:
    body, header = build_multipart(b"abc", "dir/My Deck.pptx")
    boundary = "edusync-c2-" + hashlib.sha256(b"abc").hexdigest()[:32]
    assert header == f"multipart/form-data; boundary={boundary}"
    assert (
        body
        == (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="My Deck.pptx"\r\n'
            f"Content-Type: {PPTX_CT}\r\n\r\n"
            f"abc\r\n--{boundary}--\r\n"
        ).encode()
    )


def test_multipart_deterministic_and_payload_dependent() -> None:
    assert build_multipart(b"abc", "a.pdf") == build_multipart(b"abc", "a.pdf")
    assert build_multipart(b"abc", "a.pdf")[1] != build_multipart(b"abd", "a.pdf")[1]


@pytest.mark.parametrize(
    ("filename", "expected_name", "expected_type"),
    [
        (r"C:\x\y\deck.PDF", "deck.PDF", "application/pdf"),
        ('a"b.pdf', "ab.pdf", "application/pdf"),
        ('we"ird\r\nname.pptx', "weirdname.pptx", PPTX_CT),
        ("notes.txt", "notes.txt", "application/octet-stream"),
        ("noext", "noext", "application/octet-stream"),
        ("dir/", "upload", "application/octet-stream"),
    ],
)
def test_multipart_filename_and_type(filename: str, expected_name: str, expected_type: str) -> None:
    body = build_multipart(b"x", filename)[0].decode()
    assert f'filename="{expected_name}"\r\n' in body
    assert f"Content-Type: {expected_type}\r\n" in body
    assert body.count("\r\n\r\n") == 1


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"
    seen: ClassVar[list[tuple[str, str, bytes]]] = []

    def log_message(self, *args) -> None:  # silence
        pass

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        type(self).seen.append((self.path, self.headers["Content-Type"], self.rfile.read(length)))
        mode = type(self).mode
        if mode == "slow":
            time.sleep(0.6)
        status, body = {
            "ok": (200, b'{"graph": {"nodes": []}}'),
            "slow": (200, b"{}"),
            "bad400": (400, json.dumps({"detail": "x" * 500}).encode()),
            "bad422_list": (422, json.dumps({"detail": [{"loc": "file"}]}).encode()),
            "bad415_text": (415, b"not json"),
            "bad404_list": (404, b"[1]"),
            "err500": (500, b'{"detail": "boom"}'),
            "badjson": (200, b"{not json"),
            "list": (200, b"[1, 2]"),
        }[mode]
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.mode = "ok"
    _Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/api/c1/"
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_success_posts_multipart(server: str) -> None:
    result = UrllibTransport(server, 5)(b"DATA", "deck.pdf")
    assert result == {"graph": {"nodes": []}}
    path, ctype, body = _Handler.seen[0]
    assert path == "/api/c1/lectures/graph"
    assert (body, ctype) == build_multipart(b"DATA", "deck.pdf")


def test_4xx_maps_to_422_with_truncated_detail(server: str) -> None:
    _Handler.mode = "bad400"
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport(server, 5)(b"D", "a.pdf")
    assert exc.value.status_code == 422
    assert exc.value.message == "x" * 200


@pytest.mark.parametrize("mode", ["bad422_list", "bad415_text", "bad404_list"])
def test_4xx_without_text_detail(server: str, mode: str) -> None:
    _Handler.mode = mode
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport(server, 5)(b"D", "a.pdf")
    assert (exc.value.status_code, exc.value.message) == (422, "C1 rejected the file")


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("err500", "C1 failed to build the graph"),
        ("badjson", "C1 returned an invalid response"),
        ("list", "C1 returned an invalid response"),
    ],
)
def test_502_cases(server: str, mode: str, message: str) -> None:
    _Handler.mode = mode
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport(server, 5)(b"D", "a.pdf")
    assert (exc.value.status_code, exc.value.message) == (502, message)


def test_timeout_maps_to_504(server: str) -> None:
    _Handler.mode = "slow"
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport(server, 0.1)(b"D", "a.pdf")
    assert exc.value.status_code == 504


def test_connection_refused_maps_to_502() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport(f"http://127.0.0.1:{port}", 15)(b"D", "a.pdf")
    assert (exc.value.status_code, exc.value.message) == (502, "C1 is unreachable")


def test_urlerror_timeout_maps_to_504(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*args, **kwargs):
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(c1_client.urllib.request, "urlopen", fake)
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport("http://x", 1)(b"D", "a.pdf")
    assert exc.value.status_code == 504


def test_plain_oserror_maps_to_502(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(*args, **kwargs):
        raise ConnectionResetError("reset")

    monkeypatch.setattr(c1_client.urllib.request, "urlopen", fake)
    with pytest.raises(GraphProviderError) as exc:
        UrllibTransport("http://x", 1)(b"D", "a.pdf")
    assert exc.value.status_code == 502


def test_logs_status_and_reason_only(server: str, caplog: pytest.LogCaptureFixture) -> None:
    _Handler.mode = "err500"
    with (
        caplog.at_level(logging.WARNING, logger="edusync_c2"),
        pytest.raises(GraphProviderError),
    ):
        UrllibTransport(server, 5)(b"SECRET-CONTENT", "secret-name.pdf")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "status=502" in text
    assert "SECRET" not in text
    assert "secret-name" not in text
    assert "boom" not in text
