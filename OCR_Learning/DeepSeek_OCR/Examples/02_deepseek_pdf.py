"""PDF through DeepSeek-OCR: rasterize with PyMuPDF, then infer page by page.

DeepSeek-OCR has no infer_multi — each page is an independent single-image
parse, so tables and paragraphs that span page breaks will NOT be merged
(unlike Unlimited-OCR). Per-page results land in out/page_NNNN/; the combined
markdown (pages joined with <PAGE> markers, matching the Unlimited-OCR output
shape) is written to out/result.md.

Usage:
    python 02_deepseek_pdf.py document.pdf [-o out/pdf] [--mode gundam] [--dpi 300]
"""

import argparse
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

from deepseek_engine import MODES, PROMPT_MARKDOWN, parse_image


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[str]:
    """Rasterize each page to PNG at the given DPI (same recipe as Unlimited)."""
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("out/pdf"))
    parser.add_argument("--mode", choices=MODES, default="gundam")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    pages = pdf_to_images(args.pdf, args.dpi)
    print(f"Rasterized {len(pages)} pages at {args.dpi} DPI")

    parts = []
    for i, image in enumerate(pages, 1):
        page_out = args.out / f"page_{i:04d}"
        print(f"Parsing page {i}/{len(pages)} -> {page_out}")
        parts.append(parse_image(Path(image), page_out, mode=args.mode,
                                 prompt=PROMPT_MARKDOWN))

    combined = args.out / "result.md"
    combined.write_text(
        "\n".join(f"<PAGE>\n{part.strip()}" for part in parts) + "\n",
        encoding="utf-8",
    )
    print(f"Done; per-page results under {args.out}, combined markdown in {combined}")


if __name__ == "__main__":
    main()
