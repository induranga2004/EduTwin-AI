"""Upload validation: extension, size, signature and emptiness checks."""

from pathlib import Path

from fastapi import HTTPException, UploadFile

SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".pdf", ".pptx"})
SUPPORTED_RECORDING_EXTENSIONS = frozenset(
    {".mp4", ".mp3", ".wav", ".m4a", ".webm", ".mpeg", ".mpga", ".ogg", ".flac"}
)
# Documents are capped at this size even if the global upload limit is higher.
MAX_DOCUMENT_MB = 100
_MB = 1024 * 1024


class UploadError(HTTPException):
    """HTTP error raised for invalid uploads or unreadable documents."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(status_code=status_code, detail=detail)


def document_limit_bytes(max_upload_mb: int) -> int:
    """Size limit for lecture documents."""
    return min(max_upload_mb, MAX_DOCUMENT_MB) * _MB


def recording_limit_bytes(max_upload_mb: int) -> int:
    """Size limit for audio/video recordings."""
    return max_upload_mb * _MB


async def read_upload(file: UploadFile, max_size: int) -> bytes:
    """Read an upload, enforcing the size limit (413) and non-empty content (400)."""
    content = await file.read(max_size + 1)
    if len(content) > max_size:
        raise UploadError(413, f"File exceeds the {max_size // _MB} MB limit.")
    if not content:
        raise UploadError(400, "The uploaded file is empty.")
    return content


def validate_document(filename: str | None, content: bytes) -> str:
    """Check a PDF/PPTX by name and magic bytes; return its lowercase extension."""
    if not filename:
        raise UploadError(400, "The document must have a filename.")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise UploadError(400, "Please upload a PDF or PowerPoint (.pptx) file.")
    if extension == ".pdf" and not content.startswith(b"%PDF-"):
        raise UploadError(400, "The uploaded file is not a valid PDF.")
    if extension == ".pptx" and not content.startswith(b"PK"):
        raise UploadError(400, "The uploaded file does not appear to be a valid PPTX.")
    return extension


def validate_recording(filename: str | None) -> str:
    """Check a recording's extension; return it lowercase."""
    if not filename:
        raise UploadError(400, "The recording must have a filename.")
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_RECORDING_EXTENSIONS:
        raise UploadError(
            400,
            "Unsupported recording format. Supported formats include "
            "MP4, MP3, WAV, M4A, WEBM, MPEG, OGG, and FLAC.",
        )
    return extension
