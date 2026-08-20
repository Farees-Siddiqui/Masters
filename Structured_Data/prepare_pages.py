#!/usr/bin/env python3
"""Render page 1 of every PDF and pull its native text layer as ground truth.

Runs under env_eval. Two outputs per paper, written to WORK:
  <name>_page1.png   -- what every OCR engine sees (nothing else is given to them)
  <name>.gt.txt      -- pdfplumber's extraction from the PDF text layer

Ground truth comes from the embedded text layer, never from LaTeX source.
"""
import json
import pathlib
import sys

import pdfplumber
from pdf2image import convert_from_path

ROOT = pathlib.Path(__file__).resolve().parent
PDF_DIR = ROOT / "arxiv_papers"
WORK = pathlib.Path("/tmp/ocr_bench")
DPI = 200

# pdfplumber's default x_tolerance=3 is too wide for these papers' font metrics:
# it decides adjacent glyphs are touching and emits no space, so page 1 of the
# Transformer paper comes out as "Providedproperattributionisprovided,Google...".
# Measured across the corpus, 3 -> 1.5 takes that page from 119 to 396 words and
# drops >20-char runs from 26 to 2. Using the default would have inflated every
# engine's CER by roughly ten points, uniformly and invisibly.
X_TOLERANCE = 1.5


def group_lines(words, y_tol=3.0):
    """Cluster words into visual lines by vertical position."""
    lines = []
    for w in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if lines and abs(w["top"] - lines[-1][0]) <= y_tol:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    return [(top, sorted(ws, key=lambda w: w["x0"])) for top, ws in lines]


def find_gutter(words, page_width):
    """Locate a two-column gutter, or return None for single-column pages.

    Scans candidate split positions across the middle of the page and picks the
    one crossed by the fewest words. A real gutter is crossed by almost nothing;
    a single-column page has text straddling every candidate.
    """
    if len(words) < 40:
        return None
    best, best_cross = None, None
    for frac in [0.44 + 0.01 * i for i in range(13)]:  # 0.44 .. 0.56
        x = page_width * frac
        cross = sum(1 for w in words if w["x0"] < x < w["x1"])
        left = sum(1 for w in words if w["x1"] <= x)
        right = sum(1 for w in words if w["x0"] >= x)
        if left < 0.25 * len(words) or right < 0.25 * len(words):
            continue
        if best_cross is None or cross < best_cross:
            best, best_cross = x, cross
    # Require the gutter to be genuinely clear: a handful of straddling words
    # (a spanning title, a wide figure) is fine, a hundred is not a gutter.
    if best is None or best_cross > 0.06 * len(words):
        return None
    return best


def reading_order_text(words, page_width):
    """Serialise words in human reading order.

    Single column: plain top-to-bottom. Two columns: walk lines downward; lines
    that straddle the gutter (title, full-width figure) act as barriers, and the
    column material buffered above each barrier is emitted left column first,
    then right. This is what a reader does, and what every layout-aware engine
    outputs -- unlike pdfplumber's default, which zips the columns together line
    by line and makes correct output look like 70% character error.
    """
    gutter = find_gutter(words, page_width)
    if gutter is None:
        lines = group_lines(words)
        return "\n".join(" ".join(w["text"] for w in ws) for _, ws in lines), 1

    # Classify each visual line as spanning or as one/two column lines.
    #
    # Two cases look identical if you only ask "does a word cross the gutter?":
    # a centered title whose words happen to fall either side of it, and a
    # left-column line sharing a baseline with a right-column line. The gutter
    # gap separates them -- real column whitespace is several times a word
    # space, a title's inter-word gaps are not.
    span_lines, left_lines, right_lines = [], [], []
    for top, ws in group_lines(words):
        if any(w["x0"] < gutter < w["x1"] for w in ws):
            span_lines.append((top, ws))
            continue
        lefts = [w for w in ws if w["x1"] <= gutter]
        rights = [w for w in ws if w["x0"] >= gutter]
        if not lefts or not rights:
            (left_lines if lefts else right_lines).append((top, ws))
            continue
        gutter_gap = min(w["x0"] for w in rights) - max(w["x1"] for w in lefts)
        gaps = [b["x0"] - a["x1"] for a, b in zip(ws, ws[1:]) if b["x0"] > a["x1"]]
        median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
        if gutter_gap > max(8.0, 3.0 * median_gap):
            left_lines.append((top, lefts))
            right_lines.append((top, rights))
        else:
            span_lines.append((top, ws))

    # A stray wide gap can fake a gutter on a single-column page: `adam` and
    # `vgg` (both ICLR, single column) produced 43-vs-8 and 42-vs-7 spanning-to-
    # column line splits. A genuine two-column page is the other way round
    # (BERT: 4 spanning, 54 column). If column lines do not dominate, the page is
    # single-column and plain top-to-bottom order is correct.
    if len(left_lines) + len(right_lines) <= len(span_lines):
        lines = group_lines(words)
        return "\n".join(" ".join(w["text"] for w in ws) for _, ws in lines), 1

    # Spanning lines (title, authors, full-width figures) are barriers. Column
    # text is emitted band by band between them: all of the left column in that
    # band, then all of the right.
    out = []

    def emit_band(lo, hi):
        for buf in (left_lines, right_lines):
            for top, ws in buf:
                if lo <= top < hi:
                    out.append(" ".join(w["text"] for w in ws))

    prev = float("-inf")
    for (top, ws) in span_lines:
        emit_band(prev, top)
        out.append(" ".join(w["text"] for w in ws))
        prev = top
    emit_band(prev, float("inf"))
    return "\n".join(out), 2


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {PDF_DIR}", file=sys.stderr)
        return 1

    index = {}
    for pdf in pdfs:
        name = pdf.stem
        png = WORK / f"{name}_page1.png"
        gt_path = WORK / f"{name}.gt.txt"

        with pdfplumber.open(pdf) as doc:
            page = doc.pages[0]
            gt_rowmajor = page.extract_text(x_tolerance=X_TOLERANCE) or ""
            gt_default = page.extract_text() or ""
            # Drop the rotated arXiv margin stamp. pdfplumber emits it a glyph at
            # a time bottom-to-top, so it lands in the text layer reversed --
            # "9102 yaM 42 ]LC.sc[" -- which no OCR engine can ever match. Left
            # in, it is pure error charged to every engine equally.
            words = [w for w in page.extract_words(x_tolerance=X_TOLERANCE)
                     if w.get("upright", True)]
            n_rotated = len(page.extract_words(x_tolerance=X_TOLERANCE)) - len(words)
            gt, ncols = reading_order_text(words, page.width)
            w_pt, h_pt = page.width, page.height
        gt_path.write_text(gt, encoding="utf-8")
        (WORK / f"{name}.gt_default.txt").write_text(gt_default, encoding="utf-8")
        (WORK / f"{name}.gt_rowmajor.txt").write_text(gt_rowmajor, encoding="utf-8")

        if not png.exists():
            img = convert_from_path(str(pdf), dpi=DPI, first_page=1, last_page=1)[0]
            img.save(png)
        else:
            from PIL import Image
            img = Image.open(png)

        index[name] = {
            "pdf": str(pdf),
            "image": str(png),
            "gt_text": str(gt_path),
            "gt_chars": len(gt),
            "gt_words": len(gt.split()),
            "gt_words_default_xtol": len(gt_default.split()),
            "x_tolerance": X_TOLERANCE,
            "columns": ncols,
            "rotated_words_dropped": n_rotated,
            "gt_rowmajor": str(WORK / f"{name}.gt_rowmajor.txt"),
            "page_pt": [w_pt, h_pt],
            "image_px": list(img.size),
            "dpi": DPI,
        }
        print(f"{name:12s} {img.size[0]:5d}x{img.size[1]:5d}px  gt={len(gt):6,d} chars  "
              f"{ncols}-col")

    (WORK / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nwrote {WORK / 'index.json'} ({len(index)} papers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
