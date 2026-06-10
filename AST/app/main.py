"""FastAPI entrypoint.

Run with:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .ast_builder import build_ast
from .ocr import run_ocr_on_pdf

# The layout package lives at the repo root (one level above app/).
from layout import GRANULARITIES, LayoutDetector

# The alignment package (Doc/AST <-> PDF boxes) also lives at the repo root.
from alignment import align_stream


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
        _layout_detector = LayoutDetector()  # device="auto": GPU if available, else CPU
    return _layout_detector


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/layout")
def layout_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "layout.html")


@app.get("/align")
def align_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "align.html")


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


# NOTE: PaddlePaddle inference must run on the main thread — calling it from a
# worker thread (run_in_threadpool, or a sync `def` endpoint) raises
# "Tensor holds no memory". So the handlers below run the detector synchronously,
# briefly blocking the event loop (fine for a local single-user tool).


@app.post("/api/layout/start")
async def layout_start(file: UploadFile = File(...)) -> dict:
    """Render every page to an image (fast, no OCR). Boxes are fetched per page."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    pdf_path = LAYOUT_UPLOAD_DIR / Path(file.filename).name
    pdf_path.write_bytes(pdf_bytes)

    try:
        manifest = _get_layout_detector().render_document(pdf_path, LAYOUT_OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Rendering failed: {exc}") from exc

    stem = Path(manifest["output_dir"]).name
    pages = [
        {
            "page": p["page"],
            "width": p["width"],
            "height": p["height"],
            "image_url": f"/layout-data/{stem}/{p['image']}",
        }
        for p in manifest["pages"]
    ]
    return {"doc": stem, "filename": file.filename, "page_count": manifest["page_count"], "pages": pages}


@app.get("/api/layout/page")
async def layout_page_boxes(doc: str, page: int, granularity: str = "paragraph") -> dict:
    """Run layout + OCR for ONE page at ONE granularity (cached on disk)."""
    if granularity not in GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"granularity must be one of {GRANULARITIES}, got {granularity!r}",
        )
    doc_dir = LAYOUT_OUTPUT_DIR / Path(doc).name  # sanitize against path traversal
    if not doc_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc}")

    try:
        boxes = _get_layout_detector().process_page(doc_dir, page, granularity)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Layout analysis failed: {exc}") from exc

    return {"doc": doc, "page": page, "granularity": granularity, "boxes": boxes}


# --------------------------------------------------------------------------- #
# Alignment: map AST nodes -> PDF boxes (the whiteboard's
# Alignment : Loc[Doc] -> Set[Loc[PDF]]). Two-phase like the layout tab so the
# page paints fast: /start builds the AST + page images; /compute runs layout +
# OCR per page (cached on disk) and the naive aligner.
# --------------------------------------------------------------------------- #


@app.post("/api/align/start")
async def align_start(file: UploadFile = File(...)) -> dict:
    """OCR + build AST + render page images. Persists ast.json in the doc dir."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        ocr = run_ocr_on_pdf(pdf_bytes, filename=file.filename)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OCR failed: {exc}") from exc
    ast_dict = build_ast(ocr.markdown).to_dict()

    pdf_path = LAYOUT_UPLOAD_DIR / Path(file.filename).name
    pdf_path.write_bytes(pdf_bytes)
    try:
        manifest = _get_layout_detector().render_document(pdf_path, LAYOUT_OUTPUT_DIR)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Rendering failed: {exc}") from exc

    doc_dir = Path(manifest["output_dir"])
    (doc_dir / "ast.json").write_text(json.dumps(ast_dict), encoding="utf-8")
    stem = doc_dir.name
    pages = [
        {
            "page": p["page"],
            "width": p["width"],
            "height": p["height"],
            "image_url": f"/layout-data/{stem}/{p['image']}",
        }
        for p in manifest["pages"]
    ]
    return {
        "doc": stem,
        "filename": file.filename,
        "page_count": manifest["page_count"],
        "ast": ast_dict,
        "pages": pages,
    }


@app.get("/api/align/compute")
async def align_compute(doc: str, granularity: str = "paragraph") -> dict:
    """Run layout+OCR for every page (cached), then the naive node->box aligner."""
    if granularity not in GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"granularity must be one of {GRANULARITIES}, got {granularity!r}",
        )
    doc_dir = LAYOUT_OUTPUT_DIR / Path(doc).name  # sanitize against path traversal
    if not doc_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc}")

    ast_path = doc_dir / "ast.json"
    if not ast_path.is_file():
        raise HTTPException(status_code=404, detail=f"No AST for document: {doc}")
    ast_dict = json.loads(ast_path.read_text(encoding="utf-8"))

    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    detector = _get_layout_detector()
    pages: list[dict] = []
    for p in manifest["pages"]:
        try:
            boxes = detector.process_page(doc_dir, p["page"], granularity)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Layout analysis failed: {exc}") from exc
        pages.append({"page": p["page"], "width": p["width"], "height": p["height"], "boxes": boxes})

    result = align_stream(ast_dict, pages, granularity)
    return {
        "doc": doc,
        "granularity": granularity,
        "coverage": result["coverage"],
        "alignment": result["alignment"],
        "reverse": result["reverse"],
        "pages": pages,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/layout-data", StaticFiles(directory=LAYOUT_OUTPUT_DIR), name="layout-data")
