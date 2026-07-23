"""PDF through Unlimited-OCR: rasterize at 300 DPI, then infer_multi in base mode.

infer_multi puts all pages of a chunk in one context, so tables and paragraphs
that span page breaks come out whole — don't loop infer() over pages. The
32,768-token cap covers the *entire* output, so dense documents are processed
in chunks of --chunk pages; if output still cuts off mid-parse, lower --chunk.

Usage:
    python 02_unlimited_pdf.py document.pdf [-o out/pdf] [--chunk 8]
"""

import argparse
import os
import tempfile
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: Path, dpi: int = 300) -> list[str]:
    """Rasterize each page to PNG at the given DPI (the repo's recipe)."""
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
    parser.add_argument("--chunk", type=int, default=8,
                        help="pages per infer_multi call (default 8)")
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("baidu/Unlimited-OCR",
                                              trust_remote_code=True)
    model = AutoModel.from_pretrained("baidu/Unlimited-OCR", trust_remote_code=True,
                                      use_safetensors=True,
                                      torch_dtype=torch.bfloat16)
    model = model.eval().cuda()

    pages = pdf_to_images(args.pdf)
    print(f"Rasterized {len(pages)} pages at 300 DPI")

    for start in range(0, len(pages), args.chunk):
        chunk = pages[start:start + args.chunk]
        chunk_out = args.out / f"pages_{start + 1:04d}_{start + len(chunk):04d}"
        chunk_out.mkdir(parents=True, exist_ok=True)
        print(f"Parsing pages {start + 1}-{start + len(chunk)} -> {chunk_out}")
        model.infer_multi(
            tokenizer,
            prompt="<image>Multi page parsing.",  # fixed training prompt — do not edit
            image_files=chunk,
            output_path=str(chunk_out),
            image_size=1024,  # base mode: one straight pass per page
            max_length=32768,
            no_repeat_ngram_size=35, ngram_window=1024,
            save_results=True,
        )

    print(f"Done; results under {args.out}")


if __name__ == "__main__":
    main()
