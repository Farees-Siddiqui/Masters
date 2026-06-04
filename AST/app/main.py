"""FastAPI entrypoint.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ast_builder import build_ast
from .ocr import run_ocr_on_pdf

# The layout package lives at the repo root (one level above app/).
from layout import GRANULARITIES, LayoutDetector


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
LAYOUT_OUTPUT_DIR = REPO_ROOT / "layout_output"
LAYOUT_UPLOAD_DIR = LAYOUT_OUTPUT_DIR / "_uploads"
LAYOUT_OUTPUT_DIR.mkdir(exist_ok=True)
LAYOUT_UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Document AST Viewer")

# One detector instance, reused across requests (model load is expensive). The
# PP-OCR text models load lazily on the first line/word request.
_layout_detector: LayoutDetector | None = None


def _get_layout_detector() -> LayoutDetector:
    global _layout_detector
    if _layout_detector is None:
        _layout_detector = LayoutDetector(device="cpu")
    return _layout_detector


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/layout")
def layout_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "layout.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported in v1.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        ocr = run_ocr_on_pdf(pdf_bytes, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR failed: {exc}") from exc

    ast_root = build_ast(ocr.markdown)
    return {
        "filename": file.filename,
        "page_count": ocr.page_count,
        "markdown": ocr.markdown,
        "ast": ast_root.to_dict(),
    }


@app.post("/api/layout")
async def analyze_layout(
    file: UploadFile = File(...),
    granularity: str = Form("paragraph"),
) -> dict:
    """Run layout detection at a SINGLE granularity and return per-page boxes."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if granularity not in GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"granularity must be one of {GRANULARITIES}, got {granularity!r}",
        )

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    safe_name = Path(file.filename).name
    pdf_path = LAYOUT_UPLOAD_DIR / safe_name
    pdf_path.write_bytes(pdf_bytes)

    # NOTE: PaddlePaddle inference must run on the main thread — calling it from a
    # worker thread (run_in_threadpool) raises "Tensor holds no memory". So we run
    # synchronously here, briefly blocking the event loop (fine for a local tool).
    try:
        detector = _get_layout_detector()
        result = detector.process_pdf(pdf_path, LAYOUT_OUTPUT_DIR, granularity=granularity)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Layout analysis failed: {exc}") from exc

    stem = Path(result.output_dir).name
    pages = [
        {
            "page": p.page,
            "width": p.width,
            "height": p.height,
            "image_url": f"/layout-data/{stem}/{p.image}",
            "boxes": [b.to_dict() for b in p.boxes.get(granularity, [])],
        }
        for p in result.pages
    ]
    return {
        "filename": file.filename,
        "granularity": granularity,
        "page_count": result.page_count,
        "pages": pages,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/layout-data", StaticFiles(directory=LAYOUT_OUTPUT_DIR), name="layout-data")
