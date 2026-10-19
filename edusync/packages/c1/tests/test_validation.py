import io

import pytest
from fastapi import UploadFile

from edusync_c1.validation import (
    UploadError,
    document_limit_bytes,
    read_upload,
    recording_limit_bytes,
    validate_document,
    validate_recording,
)


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [("a.pdf", b"%PDF-1.7", ".pdf"), ("B.PPTX", b"PK\x03\x04", ".pptx")],
)
def test_validate_document_ok(name, content, expected):
    assert validate_document(name, content) == expected


@pytest.mark.parametrize(
    ("name", "content"),
    [
        (None, b"%PDF-"),
        ("", b"%PDF-"),
        ("a.txt", b"hello"),
        ("a", b"hello"),
        ("a.pdf", b"not pdf"),
        ("a.pptx", b"%PDF-"),
    ],
)
def test_validate_document_rejects(name, content):
    with pytest.raises(UploadError) as err:
        validate_document(name, content)
    assert err.value.status_code == 400


@pytest.mark.parametrize("name", ["a.mp4", "a.MP3", "a.wav", "a.m4a", "a.webm", "a.ogg", "a.flac"])
def test_validate_recording_ok(name):
    assert validate_recording(name).startswith(".")


@pytest.mark.parametrize("name", [None, "", "a.exe", "noext"])
def test_validate_recording_rejects(name):
    with pytest.raises(UploadError):
        validate_recording(name)


def test_limits():
    assert document_limit_bytes(500) == 100 * 1024 * 1024
    assert document_limit_bytes(5) == 5 * 1024 * 1024
    assert recording_limit_bytes(200) == 200 * 1024 * 1024


@pytest.mark.parametrize(
    ("content", "limit", "status"),
    [(b"", 10, 400), (b"x" * 11, 10, 413)],
)
async def test_read_upload_errors(content, limit, status):
    upload = UploadFile(io.BytesIO(content), filename="f.pdf")
    with pytest.raises(UploadError) as err:
        await read_upload(upload, limit)
    assert err.value.status_code == status


async def test_read_upload_ok():
    assert await read_upload(UploadFile(io.BytesIO(b"abc"), filename="f"), 10) == b"abc"
