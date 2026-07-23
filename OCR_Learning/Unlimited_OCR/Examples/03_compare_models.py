"""Run the same image through several OCR engines and compare the outputs.

Reports per engine: wall time, output size, CER/WER against ground truth (if
--truth is given), and a pairwise similarity matrix between engine outputs.
Unlimited-OCR emits markdown while the others emit raw lines, so all text is
normalized (markup stripped, whitespace collapsed, lowercased) before scoring.

Usage:
    python 03_compare_models.py images/sample.png \
        --engines unlimited,paddle,tesseract,easyocr \
        --truth images/sample_truth.txt
"""

import argparse
import difflib
import re
from pathlib import Path

from ocr_engines import ENGINES, OCRResult, run_engine


def normalize(text: str) -> str:
    text = re.sub(r"[#*|`<>]+", " ", text)  # markdown/table markup
    return re.sub(r"\s+", " ", text).strip().lower()


def levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    return levenshtein(list(hyp), list(ref)) / max(len(ref), 1)


def wer(hyp: str, ref: str) -> float:
    ref_words = ref.split()
    return levenshtein(hyp.split(), ref_words) / max(len(ref_words), 1)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--engines", default=",".join(ENGINES),
                        help=f"comma-separated subset of: {', '.join(ENGINES)}")
    parser.add_argument("--truth", type=Path, default=None,
                        help="ground-truth text file for CER/WER")
    parser.add_argument("-o", "--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    names = [n.strip() for n in args.engines.split(",") if n.strip()]
    unknown = [n for n in names if n not in ENGINES]
    if unknown:
        parser.error(f"unknown engine(s) {unknown}; choose from {list(ENGINES)}")

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[OCRResult] = []
    for name in names:
        print(f"Running {name}...", flush=True)
        result = run_engine(name, args.image, args.out)
        if result.ok:
            (args.out / f"{name}.txt").write_text(result.text, encoding="utf-8")
        else:
            print(f"  skipped: {result.error}")
        results.append(result)

    truth = normalize(args.truth.read_text(encoding="utf-8")) if args.truth else None
    ok = [r for r in results if r.ok]

    print(f"\n{'engine':<12} {'time (s)':>9} {'chars':>7}", end="")
    if truth:
        print(f" {'CER':>7} {'WER':>7}", end="")
    print()
    for r in results:
        if not r.ok:
            print(f"{r.engine:<12} {'—':>9} {'—':>7}  ({r.error})")
            continue
        norm = normalize(r.text)
        print(f"{r.engine:<12} {r.seconds:>9.2f} {len(r.text):>7}", end="")
        if truth:
            print(f" {cer(norm, truth):>7.3f} {wer(norm, truth):>7.3f}", end="")
        print()

    if len(ok) > 1:
        print("\nPairwise similarity (normalized text):")
        print(f"{'':<12}" + "".join(f"{r.engine:>11}" for r in ok))
        for a in ok:
            row = "".join(
                f"{similarity(normalize(a.text), normalize(b.text)):>11.3f}"
                for b in ok
            )
            print(f"{a.engine:<12}{row}")

    print(f"\nRaw outputs saved to {args.out}/<engine>.txt")


if __name__ == "__main__":
    main()
