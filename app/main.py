
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from pypdf import PdfReader

app = FastAPI(
    title="EduTwin-AI C1 Backend",
    description="General-purpose lecture analysis and knowledge graph system",
    version="0.1.0",
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


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
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

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

    lecture_id = str(uuid4())
    saved_path = UPLOAD_DIR / f"{lecture_id}.pdf"

    try:
        saved_path.write_bytes(content)

        reader = PdfReader(str(saved_path))

        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Password-protected PDFs are not supported yet.",
            )

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(
                {
                    "page": page_number,
                    "text": text.strip(),
                }
            )

        full_text = "\n\n".join(
            page["text"] for page in pages if page["text"]
        )

        if not full_text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text was found. "
                    "This PDF may require OCR."
                ),
            )

        return {
            "lecture_id": lecture_id,
            "filename": file.filename,
            "page_count": len(reader.pages),
            "extracted_character_count": len(full_text),
            "pages": pages,
            "message": "PDF uploaded and text extracted successfully.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Could not process this PDF.",
        ) from exc
    finally:
        await file.close()
