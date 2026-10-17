
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader

from app.services.concept_extractor import extract_concepts
from app.services.knowledge_graph import build_knowledge_graph
from app.services.transcription import LectureTranscriber


app = FastAPI(
    title="EduTwin-AI C1 Backend",
    description=(
        "Lecture PDF processing, candidate knowledge graph generation, "
        "recording transcription, and curriculum coverage research."
    ),
    version="0.4.0",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
RECORDING_DIR = PROJECT_ROOT / "data" / "recordings"
TRANSCRIPT_DIR = PROJECT_ROOT / "data" / "processed"

for directory in (UPLOAD_DIR, RECORDING_DIR, TRANSCRIPT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_RECORDING_SIZE = 200 * 1024 * 1024  # 200 MB

ALLOWED_RECORDING_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".webm",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".flac",
}

logger = logging.getLogger(__name__)

# Load the speech model lazily when the first transcription is requested.
transcriber = LectureTranscriber(
    model_size="base",
    device="cpu",
    compute_type="int8",
)


def extract_pdf_pages(reader: PdfReader) -> list[dict]:
    """Extract text from each page of a PDF."""
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(
            {
                "page": page_number,
                "text": page_text.strip(),
            }
        )

    return pages


def combine_page_text(pages: list[dict]) -> str:
    """Combine non-empty page text into one lecture text."""
    return "\n\n".join(
        page["text"] for page in pages if page["text"]
    )


def validate_pdf_upload(file: UploadFile, content: bytes) -> None:
    """Validate the filename, size, and basic PDF signature."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF.",
        )


@app.get("/")
def home():
    return {
        "project": "EduTwin-AI",
        "component": "C1",
        "status": "running",
        "version": app.version,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/lectures/upload")
async def upload_lecture(file: UploadFile = File(...)):
    """Upload a PDF and extract its text and candidate concepts."""
    content = await file.read(MAX_FILE_SIZE + 1)

    try:
        validate_pdf_upload(file, content)

        lecture_id = str(uuid4())
        saved_path = UPLOAD_DIR / f"{lecture_id}.pdf"

        try:
            saved_path.write_bytes(content)

            reader = PdfReader(str(saved_path))

            if reader.is_encrypted:
                saved_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Password-protected PDFs are not supported yet."
                    ),
                )

            pages = extract_pdf_pages(reader)
            full_text = combine_page_text(pages)

            if not full_text.strip():
                saved_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No extractable text was found. "
                        "This PDF may require OCR."
                    ),
                )

            concepts = extract_concepts(full_text)

            return {
                "lecture_id": lecture_id,
                "filename": file.filename,
                "page_count": len(pages),
                "extracted_character_count": len(full_text),
                "concept_count": len(concepts),
                "concepts": concepts,
                "pages": pages,
                "message": (
                    "PDF uploaded, text extracted, "
                    "and candidate concepts identified."
                ),
            }

        except HTTPException:
            raise
        except Exception:
            saved_path.unlink(missing_ok=True)
            raise

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PDF upload failed")
        raise HTTPException(
            status_code=400,
            detail="Could not process this PDF.",
        ) from exc
    finally:
        await file.close()


@app.post("/api/lectures/graph")
async def generate_lecture_graph(file: UploadFile = File(...)):
    """Generate a candidate knowledge graph from an uploaded lecture PDF."""
    try:
        content = await file.read(MAX_FILE_SIZE + 1)
        validate_pdf_upload(file, content)

        reader = PdfReader(BytesIO(content))

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported yet.",
            )

        pages = extract_pdf_pages(reader)
        full_text = combine_page_text(pages)

        if not full_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text was found. "
                    "This PDF may require OCR."
                ),
            )

        concepts = extract_concepts(full_text)
        graph = build_knowledge_graph(pages, concepts)

        return {
            "lecture_id": str(uuid4()),
            "filename": file.filename,
            "page_count": len(pages),
            "concept_count": len(concepts),
            "concepts": concepts,
            "graph": graph,
            "message": (
                "Candidate knowledge graph generated successfully."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Knowledge graph generation failed")
        raise HTTPException(
            status_code=500,
            detail="Could not generate the knowledge graph from this PDF.",
        ) from exc
    finally:
        await file.close()


@app.post("/api/lectures/transcribe")
async def transcribe_lecture(
    file: UploadFile = File(...),
    language: str = "en",
):
    """
    Upload an audio/video recording and generate timestamped speech.

    The returned transcript is automatically generated. A separate
    transcript file is not required.
    """
    saved_path = None

    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="The recording must have a filename.",
            )

        extension = Path(file.filename).suffix.lower()

        if extension not in ALLOWED_RECORDING_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported recording format. Use MP4, MP3, WAV, "
                    "M4A, WEBM, OGG, FLAC, or a supported audio format."
                ),
            )

        content = await file.read(MAX_RECORDING_SIZE + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded recording is empty.",
            )

        if len(content) > MAX_RECORDING_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Recording exceeds the 200 MB limit.",
            )

        recording_id = str(uuid4())
        saved_path = RECORDING_DIR / f"{recording_id}{extension}"
        saved_path.write_bytes(content)

        # Transcription is CPU-intensive, so run it in a worker thread.
        from starlette.concurrency import run_in_threadpool

        transcript_data = await run_in_threadpool(
            transcriber.transcribe,
            saved_path,
            language,
        )

        transcript_data["recording_id"] = recording_id

        transcript_path = (
            TRANSCRIPT_DIR / f"{recording_id}_transcript.json"
        )

        await run_in_threadpool(
            transcriber.save_transcript,
            transcript_data,
            transcript_path,
        )

        return {
            "recording_id": recording_id,
            "filename": file.filename,
            "transcript_path": str(transcript_path),
            "language": transcript_data["language"],
            "segment_count": transcript_data["segment_count"],
            "transcript": transcript_data["transcript"],
            "segments": transcript_data["segments"],
            "message": (
                "Recording transcribed successfully. "
                "Timestamped segments and transcript JSON were generated."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Recording transcription failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not transcribe the recording. "
                "Check the server terminal for the underlying error."
            ),
        ) from exc
    finally:
        await file.close()
