"""A pinned, reproducible corpus of born-digital PDFs for the real benchmark.

Why arXiv rather than a packaged layout dataset
-----------------------------------------------
The text-layer oracle (:mod:`benchmark.oracle`) needs a **born-digital PDF** —
it reads the glyph coordinates the document itself records. The obvious
candidates don't supply that: DocLayNet and OmniDocBench as published on
HuggingFace ship *rasterised page images* plus annotations, and a PNG has no
text layer. DocLayNet's PDFs exist only in a separate ~7.5 GB archive on IBM's
CDN (worth adding later as a second corpus — it is the cross-domain story).

arXiv gives real, multi-page, born-digital PDFs for free, and pinning
``id + version`` makes a run reproducible from this file alone. Papers are
chosen across single- and two-column venues to get what layout variety a single
domain allows.

The honest limitation: **this is one domain.** Every document here is a
scientific paper. It says nothing about financial reports, forms or manuals —
which is exactly the gap DocLayNet closes.

Usage::

    python -m benchmark fetch          # download + OCR + layout (slow, cached)
    python -m benchmark real --all     # score every built doc
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus"
LAYOUT_OUTPUT = REPO_ROOT / "layout_output"

_UA = "Mozilla/5.0 (compatible; AST-alignment-benchmark/1.0; research)"


def hms(seconds: float) -> str:
    """Format a duration as m:ss / h:mm:ss."""
    s = int(round(seconds))
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


class Progress:
    """Page-level progress whose ETA is *measured*, never guessed.

    The rate is computed only from pages this run actually processed. Cached
    pages finish instantly, so folding them in would inflate throughput and
    produce a confidently wrong ETA — they are excluded from both the numerator
    and the remaining-work denominator.

    Before any page completes there is no measurement, so the ETA reports
    ``—`` rather than a fabricated number.
    """

    def __init__(self, todo_pages: int, total_docs: int) -> None:
        self.t0 = time.perf_counter()
        self.todo_pages = todo_pages
        self.total_docs = total_docs
        self.done_pages = 0
        self.done_docs = 0

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    @property
    def secs_per_page(self) -> float | None:
        """Measured seconds/page over this run, or None until a page lands."""
        return self.elapsed / self.done_pages if self.done_pages else None

    @property
    def eta(self) -> float | None:
        """Remaining seconds, extrapolated from measured throughput."""
        rate = self.secs_per_page
        if rate is None:
            return None
        return max(self.todo_pages - self.done_pages, 0) * rate

    def page_done(self) -> None:
        self.done_pages += 1

    def line(self, prefix: str) -> str:
        rate = self.secs_per_page
        eta = self.eta
        parts = [
            f"[{self.done_pages}/{self.todo_pages} pages]",
            prefix,
            f"elapsed {hms(self.elapsed)}",
            f"{rate:.1f}s/page" if rate is not None else "—s/page",
            f"eta {hms(eta)}" if eta is not None else "eta —",
        ]
        return "  ".join(parts)

    def log(self, prefix: str) -> None:
        print(self.line(prefix), flush=True)


def _pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    return len(pdfium.PdfDocument(str(path)))


def _pages_todo(doc: Doc, granularity: str) -> int:
    """Pages of this doc still needing layout+OCR (i.e. not already cached)."""
    src = pdf_path(doc)
    if not src.is_file():
        return 0
    total = _pdf_page_count(src)
    doc_dir = LAYOUT_OUTPUT / doc.name
    cached = sum(
        1 for p in range(1, total + 1) if (doc_dir / f"page{p}" / f"{granularity}.json").is_file()
    )
    return total - cached


@dataclass(frozen=True)
class Doc:
    name: str  # becomes the layout_output/<name> cache dir
    arxiv_id: str  # pinned WITH version so the bytes are stable
    note: str


# Pinned by id+version: arXiv PDFs are immutable per version, so a rerun months
# from now scores the same bytes. Mixed venues => mixed column counts.
ARXIV_CORPUS: list[Doc] = [
    Doc("resnet", "1512.03385v1", "CVPR, two-column"),
    Doc("attention", "1706.03762v7", "NeurIPS, single-column"),
    Doc("bert", "1810.04805v2", "NAACL, two-column"),
    Doc("vgg", "1409.1556v6", "ICLR, single-column"),
    Doc("faster-rcnn", "1506.01497v3", "NeurIPS, two-column"),
    Doc("adam", "1412.6980v9", "ICLR, single-column"),
    Doc("vit", "2010.11929v2", "ICLR, single-column"),
    Doc("unet", "1505.04597v1", "MICCAI, two-column"),
    Doc("densenet", "1608.06993v5", "CVPR, two-column"),
    Doc("batchnorm", "1502.03167v3", "ICML, two-column"),
]

CORPORA: dict[str, list[Doc]] = {"arxiv": ARXIV_CORPUS}


def pdf_path(doc: Doc) -> Path:
    return CORPUS_DIR / f"{doc.name}.pdf"


def fetch_pdf(doc: Doc, delay: float = 3.0) -> Path:
    """Download one arXiv PDF, skipping if already present.

    A deliberate delay between downloads keeps this polite to arXiv; the whole
    corpus is fetched once and then lives on disk.
    """
    dest = pdf_path(doc)
    if dest.is_file() and dest.stat().st_size > 10_000:
        print(f"  {doc.name:<14} cached  ({dest.stat().st_size // 1024} KB)")
        return dest

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/pdf/{doc.arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=90) as r:
        data = r.read()
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{doc.name}: not a PDF (got {data[:20]!r})")
    dest.write_bytes(data)
    print(f"  {doc.name:<14} downloaded ({len(data) // 1024} KB)  {doc.arxiv_id}")
    time.sleep(delay)
    return dest


def is_built(doc: Doc) -> bool:
    """Has this doc got an AST + a full set of paragraph boxes cached?"""
    d = LAYOUT_OUTPUT / doc.name
    man = d / "manifest.json"
    if not (d / "ast.json").is_file() or not man.is_file():
        return False
    meta = json.loads(man.read_text(encoding="utf-8"))
    return all((d / p["dir"] / "paragraph.json").is_file() for p in meta["pages"])


def build_doc(
    doc: Doc,
    granularity: str = "paragraph",
    progress: Progress | None = None,
    tag: str = "",
) -> dict:
    """Run the real pipeline for one doc: Mistral AST + render + Paddle layout.

    Everything is cached on disk, so re-running is cheap and a partially built
    doc resumes rather than restarting. Imports are local because pulling in
    Paddle/Mistral costs seconds and `fetch` alone shouldn't pay it.
    """
    from app.ast_builder import build_ast
    from app.ocr import run_ocr_on_pdf
    from layout.detector import LayoutDetector

    src = pdf_path(doc)
    doc_dir = LAYOUT_OUTPUT / doc.name
    detector = LayoutDetector()

    # 1. Mistral OCR -> AST (skip if we already have one).
    ast_file = doc_dir / "ast.json"
    if not ast_file.is_file():
        t0 = time.perf_counter()
        print(f"  {tag}{doc.name:<14} OCR -> AST (Mistral)…", flush=True)
        ocr = run_ocr_on_pdf(src.read_bytes(), filename=src.name)
        ast_dict = build_ast(ocr.markdown).to_dict()
        doc_dir.mkdir(parents=True, exist_ok=True)
        ast_file.write_text(json.dumps(ast_dict), encoding="utf-8")
        print(f"  {tag}{doc.name:<14} AST built: {ocr.page_count}p in {hms(time.perf_counter()-t0)}", flush=True)
    else:
        print(f"  {tag}{doc.name:<14} AST cached", flush=True)

    # 2. Render pages.
    if not (doc_dir / "manifest.json").is_file():
        t0 = time.perf_counter()
        detector.render_document(src, LAYOUT_OUTPUT)
        print(f"  {tag}{doc.name:<14} rendered in {hms(time.perf_counter()-t0)}", flush=True)
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))

    # 3. Layout + OCR per page (cached per page, so this resumes).
    todo = [p for p in manifest["pages"] if not (doc_dir / p["dir"] / f"{granularity}.json").is_file()]
    for p in todo:
        t0 = time.perf_counter()
        detector.process_page(doc_dir, p["page"], granularity)
        took = time.perf_counter() - t0
        if progress is not None:
            progress.page_done()
            progress.log(f"{doc.name:<14} page {p['page']:>2}/{manifest['page_count']:<2} ({took:.1f}s)")
        else:
            print(f"  {doc.name:<14} page {p['page']}/{manifest['page_count']} ({took:.1f}s)", flush=True)

    return manifest


def build_corpus(docs: list[Doc], granularity: str = "paragraph") -> list[dict]:
    """Fetch + build every doc, logging measured progress. Failures are skipped."""
    t_start = time.perf_counter()

    # Fetch first so page counts (and therefore the work total) are knowable
    # before any expensive inference starts.
    print(f"\n--- fetching {len(docs)} PDFs ---", flush=True)
    fetched: list[Doc] = []
    out: list[dict] = []
    for doc in docs:
        try:
            fetch_pdf(doc)
            fetched.append(doc)
        except Exception as exc:
            print(f"  {doc.name:<14} FETCH FAILED: {exc}", flush=True)
            out.append({"name": doc.name, "ok": False, "error": f"fetch: {exc}"})

    todo_pages = sum(_pages_todo(d, granularity) for d in fetched)
    total_pages = sum(_pdf_page_count(pdf_path(d)) for d in fetched)
    cached_pages = total_pages - todo_pages
    print(
        f"\n--- building {len(fetched)} docs: {total_pages} pages "
        f"({cached_pages} cached, {todo_pages} to process) ---",
        flush=True,
    )

    progress = Progress(todo_pages, len(fetched))
    for i, doc in enumerate(fetched, 1):
        tag = f"[{i}/{len(fetched)}] "
        try:
            manifest = build_doc(doc, granularity, progress=progress, tag=tag)
            progress.done_docs += 1
            out.append({
                "name": doc.name,
                "ok": True,
                "arxiv_id": doc.arxiv_id,
                "note": doc.note,
                "pages": manifest["page_count"],
            })
            print(f"  {tag}{doc.name:<14} READY ({manifest['page_count']} pages)  "
                  f"[docs {progress.done_docs}/{len(fetched)}]", flush=True)
        except Exception as exc:
            print(f"  {tag}{doc.name:<14} BUILD FAILED: {type(exc).__name__}: {exc}", flush=True)
            out.append({"name": doc.name, "ok": False, "error": f"build: {exc}"})

    total = time.perf_counter() - t_start
    rate = progress.secs_per_page
    print(
        f"\ntotal wall time {hms(total)}"
        + (f" · {rate:.1f}s/page measured over {progress.done_pages} processed pages" if rate else "")
        + (f" · {cached_pages} pages served from cache" if cached_pages else ""),
        flush=True,
    )
    return out


def built_docs(docs: list[Doc]) -> list[Doc]:
    """The subset that is actually ready to score."""
    return [d for d in docs if is_built(d) and pdf_path(d).is_file()]
