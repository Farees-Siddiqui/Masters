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
from alignment import align_similarity, align_stream
from alignment.corpus_index import corpus_map, corpus_trace
from alignment.doc_diff import diff_documents
from alignment.search import search_nodes

from .annotate import router as annotate_router
from .qa import answer_question

# Selectable node->box aligners for /api/align/compute (?method=...).
ALIGN_METHODS = {
    "stream": align_stream,        # positional char-stream diff (exact text)
    "similarity": align_similarity,  # semantic embedding cosine similarity
}


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "static"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
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


@app.get("/benchmark")
def benchmark_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "benchmark.html")


@app.get("/diff")
def diff_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "diff.html")


@app.get("/corpus")
def corpus_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "corpus.html")


@app.get("/annotate")
def annotate_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "annotate.html")


@app.get("/api/benchmark")
def benchmark_data() -> dict:
    """Serve both benchmark modes for the dashboard.

    ``regimes`` — the synthetic noise sweep (``python -m benchmark synthetic
    [--shuffle]``): reading order intact vs. shuffled, each a list of
    per-(method, noise) rows.

    ``real`` — the real-document run against PDF text-layer gold (``python -m
    benchmark real``), or ``None`` if it hasn't been generated yet.
    """
    regimes = []
    for key, label, fname in [
        ("intact", "Reading order intact", "results_paragraph.json"),
        ("shuffled", "Reading order shuffled", "results_shuffled.json"),
    ]:
        path = BENCHMARK_DIR / fname
        rows = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        regimes.append({"key": key, "label": label, "file": fname, "rows": rows})

    real_path = REPO_ROOT / "results_real.json"
    real = json.loads(real_path.read_text(encoding="utf-8")) if real_path.is_file() else None

    return {"regimes": regimes, "real": real}


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
async def align_compute(
    doc: str,
    granularity: str = "paragraph",
    method: str = "stream",
    threshold: float = 0.5,
) -> dict:
    """Run layout+OCR for every page (cached), then a node->box aligner.

    ``method`` selects the aligner: ``stream`` (positional char-stream diff) or
    ``similarity`` (semantic embedding cosine, honouring ``threshold``).
    """
    if granularity not in GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"granularity must be one of {GRANULARITIES}, got {granularity!r}",
        )
    if method not in ALIGN_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"method must be one of {sorted(ALIGN_METHODS)}, got {method!r}",
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

    if method == "similarity":
        try:
            result = align_similarity(ast_dict, pages, granularity, threshold=threshold)
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="similarity method needs sentence-transformers: pip install sentence-transformers",
            ) from exc
    else:
        result = align_stream(ast_dict, pages, granularity)
    return {
        "doc": doc,
        "granularity": granularity,
        "method": method,
        "coverage": result["coverage"],
        "alignment": result["alignment"],
        "reverse": result["reverse"],
        # similarity ships per-pair scores (+ floor) so the UI slider re-filters live.
        "scores": result.get("scores"),
        "score_floor": result.get("score_floor"),
        # stream ships per-word boxes for word/sentence selection on the Doc side.
        "tokens": result.get("tokens"),
        "pages": pages,
    }


def _load_doc_ast(doc: str) -> dict:
    """Load a previously-aligned document's persisted AST (or 404)."""
    doc_dir = LAYOUT_OUTPUT_DIR / Path(doc).name  # sanitize against path traversal
    ast_path = doc_dir / "ast.json"
    if not ast_path.is_file():
        raise HTTPException(status_code=404, detail=f"No AST for document: {doc}")
    return json.loads(ast_path.read_text(encoding="utf-8"))


def _first_title(node: dict) -> str | None:
    """The document's first section heading, used as a human-readable title."""
    if node.get("type") == "section":
        title = (node.get("attribs") or {}).get("title")
        if title:
            return title
    for child in node.get("children") or []:
        found = _first_title(child)
        if found:
            return found
    return None


def _corpus_doc_dirs() -> list[Path]:
    """Every processed document on disk (has both ast.json and manifest.json)."""
    dirs = []
    for d in sorted(LAYOUT_OUTPUT_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("_"):
            if (d / "ast.json").is_file() and (d / "manifest.json").is_file():
                dirs.append(d)
    return dirs


def _corpus_docs_and_signature() -> tuple[list[dict], str]:
    """Load every processed doc's AST + a cache signature (names + mtimes)."""
    docs: list[dict] = []
    sig_parts: list[str] = []
    for d in _corpus_doc_dirs():
        ast = json.loads((d / "ast.json").read_text(encoding="utf-8"))
        docs.append({"doc": d.name, "title": _first_title(ast) or d.name, "ast": ast})
        sig_parts.append(f"{d.name}:{(d / 'ast.json').stat().st_mtime_ns}")
    return docs, "|".join(sig_parts)


@app.get("/api/corpus/list")
def corpus_list() -> dict:
    """List the processed documents available for cross-document search."""
    docs = []
    for d in _corpus_doc_dirs():
        ast = json.loads((d / "ast.json").read_text(encoding="utf-8"))
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        docs.append({"doc": d.name, "title": _first_title(ast) or d.name, "page_count": manifest.get("page_count")})
    return {"docs": docs}


@app.get("/api/corpus/search")
def corpus_search(q: str, top_k: int = 12, per_doc: int = 5) -> dict:
    """Semantic search across every processed document; globally ranked results.

    Each result carries its ``doc`` and ``title`` so the frontend can open the
    owning document and highlight the node.
    """
    results: list[dict] = []
    for d in _corpus_doc_dirs():
        ast = json.loads((d / "ast.json").read_text(encoding="utf-8"))
        title = _first_title(ast) or d.name
        for r in search_nodes(ast, q, top_k=per_doc):
            results.append({**r, "doc": d.name, "title": title})
    results.sort(key=lambda r: -r["score"])
    return {"query": q, "results": results[:top_k]}


@app.get("/api/corpus/map")
def corpus_map_route(max_points: int = 1500, n_clusters: int = 8) -> dict:
    """2D semantic map of the whole corpus (t-SNE + KMeans themes)."""
    docs, signature = _corpus_docs_and_signature()
    return corpus_map(docs, signature, max_points=max_points, n_clusters=n_clusters)


@app.get("/api/corpus/trace")
def corpus_trace_route(q: str, min_score: float = 0.35) -> dict:
    """Every passage across the corpus matching concept ``q``, grouped by document."""
    docs, signature = _corpus_docs_and_signature()
    return corpus_trace(docs, signature, q, min_score=min_score)


@app.get("/api/diff")
def doc_diff(a: str, b: str, threshold: float = 0.6) -> dict:
    """Structural diff of two processed documents (added/removed/modified)."""
    ast_a = _load_doc_ast(a)
    ast_b = _load_doc_ast(b)
    try:
        result = diff_documents(ast_a, ast_b, threshold=threshold)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="diff needs sentence-transformers: pip install sentence-transformers",
        ) from exc
    return {"a": a, "b": b, **result}


@app.get("/api/align/load")
def align_load(doc: str) -> dict:
    """Open an already-processed document (no OCR) — same shape as /align/start."""
    doc_dir = LAYOUT_OUTPUT_DIR / Path(doc).name  # sanitize against path traversal
    ast_path = doc_dir / "ast.json"
    man_path = doc_dir / "manifest.json"
    if not ast_path.is_file() or not man_path.is_file():
        raise HTTPException(status_code=404, detail=f"Unknown document: {doc}")
    ast = json.loads(ast_path.read_text(encoding="utf-8"))
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
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
        "filename": _first_title(ast) or stem,
        "page_count": manifest["page_count"],
        "ast": ast,
        "pages": pages,
    }


@app.get("/api/align/search")
async def align_search(doc: str, q: str, top_k: int = 8) -> dict:
    """Semantic search over the AST: rank nodes by similarity to ``q``.

    Returns ``{query, results: [{node_id, score, snippet}]}``. The frontend maps
    each ``node_id`` to PDF boxes via the alignment already on screen.
    """
    ast_dict = _load_doc_ast(doc)
    try:
        results = search_nodes(ast_dict, q, top_k=top_k)
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="search needs sentence-transformers: pip install sentence-transformers",
        ) from exc
    return {"query": q, "results": results}


@app.get("/api/align/ask")
async def align_ask(doc: str, q: str) -> dict:
    """Grounded Q&A: answer ``q`` from the AST with inline ``[node_id]`` citations."""
    ast_dict = _load_doc_ast(doc)
    try:
        result = answer_question(ast_dict, q)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Q&A failed: {exc}") from exc
    return {"query": q, **result}


app.include_router(annotate_router)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/layout-data", StaticFiles(directory=LAYOUT_OUTPUT_DIR), name="layout-data")
