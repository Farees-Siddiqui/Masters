"""Single image through Unlimited-OCR in gundam mode.

Gundam mode (base_size=1024, image_size=640, crop_mode=True) adds cropped local
views on top of the global one — the right choice for a single, possibly dense
page. For multi-page documents use 02_unlimited_pdf.py instead.

Usage:
    python 01_unlimited_single_image.py images/sample.png [-o out/single]
"""

import argparse
from pathlib import Path

from ocr_engines import run_unlimited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("out/single"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    text = run_unlimited(args.image, args.out)

    print(text)
    print(f"\n--- {len(text)} chars; full results saved under {args.out} ---")


if __name__ == "__main__":
    main()
