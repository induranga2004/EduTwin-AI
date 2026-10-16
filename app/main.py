
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader

from app.services.concept_extractor import extract_concepts
from app.services.knowledge_graph import build_knowledge_graph


app = FastAPI(
    title="EduTwin-AI C1 Backend",
    description=(
        "General-purpose lecture processing, concept extraction, "
        "knowledge graph generation, and curriculum coverage research."
    ),
    version="0.3.1",
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


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
        print(f"PDF upload error: {exc!r}")
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

        # PdfReader needs a file-like object for in-memory PDF bytes.
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
        # Details appear in the VS Code server terminal for debugging.
        print(f"Knowledge graph generation error: {exc!r}")
        raise HTTPException(
            status_code=500,
            detail="Could not generate the knowledge graph from this PDF.",
        ) from exc
    finally:
        await file.close()
