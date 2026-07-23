"""Single image through DeepSeek-OCR in gundam mode.

Gundam mode (base_size=1024, image_size=640, crop_mode=True) tiles the page
into local crops on top of the global view — the right choice for a single,
possibly dense page. Use --prompt free for plain text without layout markup.

Usage:
    python 01_deepseek_single_image.py ../../Unlimited_OCR/Examples/images/sample.png
    python 01_deepseek_single_image.py page.png --mode base --prompt free
"""

import argparse
from pathlib import Path

from deepseek_engine import MODES, PROMPT_FREE_OCR, PROMPT_MARKDOWN, parse_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("out/single"))
    parser.add_argument("--mode", choices=MODES, default="gundam")
    parser.add_argument("--prompt", choices=["markdown", "free"], default="markdown",
                        help="markdown = layout + grounding boxes; free = plain text")
    args = parser.parse_args()

    prompt = PROMPT_MARKDOWN if args.prompt == "markdown" else PROMPT_FREE_OCR
    text = parse_image(args.image, args.out, mode=args.mode, prompt=prompt)

    print(text)
    print(f"\n--- {len(text)} chars; full results saved under {args.out} ---")


if __name__ == "__main__":
    main()
