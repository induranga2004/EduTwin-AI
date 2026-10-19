import json

import pytest

from edusync_c1 import COMPONENT

UPLOAD = "/api/c1/lectures/upload"
GRAPH = "/api/c1/lectures/graph"
TRANSCRIBE = "/api/c1/lectures/transcribe"
PDF = "application/pdf"
OVER_LIMIT = b"%PDF-" + b"x" * (1024 * 1024 + 1)  # settings fixture caps uploads at 1 MB


def test_component_contract():
    assert COMPONENT.name == "c1"
    assert COMPONENT.prefix == "/api/c1"


def test_health(client):
    response = client.get("/api/c1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "component": "c1"}


def test_upload_pdf(client, lecture_pdf, settings):
    response = client.post(UPLOAD, files={"file": ("lecture.pdf", lecture_pdf, PDF)})
    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 1
    assert body["concept_count"] == len(body["concepts"]) > 0
    assert "Gradient Descent" in body["pages"][0]["text"]
    assert (settings.data_dir / "c1" / "uploads" / f"{body['lecture_id']}.pdf").is_file()
    assert str(settings.data_dir) not in response.text


@pytest.mark.parametrize(
    ("url", "files", "status"),
    [
        (UPLOAD, {"file": ("a.txt", b"hello", "text/plain")}, 400),
        (UPLOAD, {"file": ("a.pdf", b"not a pdf", PDF)}, 400),
        (UPLOAD, {"file": ("a.pdf", b"", PDF)}, 400),
        (UPLOAD, {"file": ("a.pdf", OVER_LIMIT, PDF)}, 413),
        (UPLOAD, {"file": ("a.pdf", "BLANK", PDF)}, 422),
        (UPLOAD, {"file": ("a.pdf", b"%PDF-1.4 junk", PDF)}, 422),
        (GRAPH, {"file": ("a.pdf", "BLANK", PDF)}, 422),
        (GRAPH, {"file": ("a.docx", b"PK", "application/zip")}, 400),
    ],
)
def test_upload_error_statuses(client, blank_pdf, url, files, status):
    files = {k: (n, blank_pdf if c == "BLANK" else c, t) for k, (n, c, t) in files.items()}
    assert client.post(url, files=files).status_code == status


def test_upload_missing_file_is_422(client):
    assert client.post(UPLOAD).status_code == 422


def test_upload_rejects_pptx(client, lecture_pptx):
    response = client.post(UPLOAD, files={"file": ("d.pptx", lecture_pptx, "application/zip")})
    assert response.status_code == 400
    assert "PDF files only" in response.json()["detail"]


def test_graph_pdf_and_pptx(client, lecture_pdf, lecture_pptx):
    pdf = client.post(GRAPH, files={"file": ("l.pdf", lecture_pdf, PDF)}).json()
    assert pdf["graph"]["node_count"] > 0
    assert pdf["graph"]["edge_count"] > 0
    pptx = client.post(GRAPH, files={"file": ("d.pptx", lecture_pptx, "application/zip")})
    assert pptx.status_code == 200
    assert pptx.json()["page_count"] == 2


def test_transcribe(client, lecture_pdf, fake_transcriber, settings):
    response = client.post(
        TRANSCRIBE + "?language=en",
        files={
            "file": ("rec.mp3", b"audio-bytes", "audio/mpeg"),
            "document_file": ("lecture.pdf", lecture_pdf, PDF),
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["segment_count"] == 3
    assert body["transcript"].startswith("Today we cover")
    assert body["language"] == "en"
    assert len(body["graph_timeline"]["timeline"]) == 3
    assert "transcript_path" not in body
    assert str(settings.data_dir) not in response.text
    path, language = fake_transcriber.calls[0]
    assert path.suffix == ".mp3"
    assert language == "en"
    saved = settings.data_dir / "c1" / "processed" / f"{body['recording_id']}_transcript.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["recording_id"] == body["recording_id"]


def _transcribe(client, recording=("rec.wav", b"x", "audio/wav"), document=None, query=""):
    files = {"file": recording}
    if document is not None:
        files["document_file"] = document
    return client.post(TRANSCRIBE + query, files=files)


def test_transcribe_errors(client, lecture_pdf, blank_pdf):
    doc = ("l.pdf", lecture_pdf, PDF)
    assert _transcribe(client, ("rec.exe", b"x", "x"), doc).status_code == 400
    assert _transcribe(client, ("rec.wav", b"", "x"), doc).status_code == 400
    assert _transcribe(client, document=None).status_code == 422
    assert _transcribe(client, document=doc, query="?language=x").status_code == 422
    assert _transcribe(client, ("rec.wav", OVER_LIMIT, "x"), doc).status_code == 413
    assert _transcribe(client, document=("l.pdf", blank_pdf, PDF)).status_code == 422


def test_transcribe_no_segments_is_422(client, fake_transcriber, lecture_pdf):
    fake_transcriber.result = {"text": "", "segments": []}
    response = _transcribe(client, document=("l.pdf", lecture_pdf, PDF))
    assert response.status_code == 422


def test_transcribe_backend_failure_is_500(client, fake_transcriber, lecture_pdf):
    fake_transcriber.result = 42  # unsupported result type
    response = _transcribe(client, document=("l.pdf", lecture_pdf, PDF))
    assert response.status_code == 500
    assert "Could not transcribe" in response.json()["detail"]


@pytest.mark.parametrize("url", [UPLOAD, GRAPH])
def test_unexpected_failure_is_500_without_details(client, lecture_pdf, monkeypatch, url):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr("edusync_c1.api.extract_concepts", boom)
    response = client.post(url, files={"file": ("l.pdf", lecture_pdf, PDF)})
    assert response.status_code == 500
    assert "secret" not in response.text
