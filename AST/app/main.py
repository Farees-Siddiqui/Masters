"""FastAPI entrypoint.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ast_builder import build_ast
from .ocr import run_ocr_on_pdf


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"

app = FastAPI(title="Document AST Viewer")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
