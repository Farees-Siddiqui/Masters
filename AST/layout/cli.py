"""Command-line interface for layout detection.

Usage:
    python -m layout INPUT.pdf -o OUTPUT_DIR [--dpi 200] [--device cpu]

All artifacts for a PDF land under ``OUTPUT_DIR/<pdf_stem>/``, one folder per
page (``page1/``, ``page2/`` ...). See :mod:`layout.detector` for the layout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .detector import (
    ALL_GRANULARITY,
    DEFAULT_DEVICE,
    DEFAULT_DPI,
    DEFAULT_GRANULARITY,
    DEFAULT_MODEL,
    GRANULARITIES,
    detect_layout,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m layout",
        description="Detect document layout in a PDF using PP-DocLayoutV3.",
    )
    parser.add_argument("pdf", help="Path to the input PDF document.")
    parser.add_argument(
        "-o",
        "--output",
        default="output",
        help="Output directory. Results go in <output>/<pdf_name>/ (default: output).",
    )
    parser.add_argument(
        "-g",
        "--granularity",
        choices=(*GRANULARITIES, ALL_GRANULARITY),
        default=DEFAULT_GRANULARITY,
        help=(
            "BBox granularity to emit: 'paragraph' = PP-DocLayoutV3 regions, "
            "'line'/'word' = PP-OCRv5 text boxes, 'all' = every level "
            f"(default: {DEFAULT_GRANULARITY}). Each level writes "
            "<level>.png (drawn boxes) and <level>.json (bbox+label+text)."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render resolution for PDF pages (default: {DEFAULT_DPI}).",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Inference device, e.g. cpu / gpu (default: {DEFAULT_DEVICE}).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"PaddleOCR layout model name (default: {DEFAULT_MODEL}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    pdf_path = Path(args.pdf).expanduser()
    if not pdf_path.is_file():
        print(f"error: PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"error: expected a .pdf file, got: {pdf_path.name}", file=sys.stderr)
        return 1

    print(
        f"Loading model '{args.model}' (granularity={args.granularity}) "
        f"on device '{args.device}' ...",
        flush=True,
    )
    result = detect_layout(
        pdf_path,
        args.output,
        dpi=args.dpi,
        model_name=args.model,
        device=args.device,
        granularity=args.granularity,
    )

    print(
        f"Done: {result.page_count} page(s), granularities={result.granularities} "
        f"-> {result.output_dir}",
        flush=True,
    )
    for page in result.pages:
        parts = []
        for g in result.granularities:
            boxes = page.boxes.get(g, [])
            with_text = sum(1 for b in boxes if b.text)
            parts.append(f"{g}={len(boxes)} ({with_text} w/ text)")
        print(f"  {page.dir}: " + ", ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
