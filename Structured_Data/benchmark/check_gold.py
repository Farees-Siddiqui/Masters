"""Hold the gold set to the same grounding rule the extractor is held to.

`schema.py` says a value is "a span that appears in the text". A gold record
whose value is a sentence the gold author composed does not satisfy that, and
no extractor can ever match it — so it measures the author, not the pipeline.

This checks every gold value with `extract._locate` against the same OCR text
the pipeline consumed, in the same markup-stripped view. A value that cannot be
located is not gold; it is commentary.

Usage:
    python benchmark/check_gold.py benchmark/gold_resnet.json \\
           --text samples/resnet.mistral.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extract import _locate, _norm_index  # noqa: E402

MAX_WORDS = 6  # the pipeline's longest emitted value is 6 words


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gold", type=Path)
    ap.add_argument("--text", type=Path, default=Path("samples/resnet.mistral.md"))
    ap.add_argument("--list", action="store_true", help="print every failing record")
    args = ap.parse_args()

    text = args.text.read_text(encoding="utf-8")
    norm, starts, ends = _norm_index(text)
    records = json.loads(args.gold.read_text(encoding="utf-8"))["records"]

    located, verbose, both_ok = [], [], []
    for r in records:
        ok = _locate(r["value"], norm, starts, ends) is not None
        short = len(r["value"].split()) <= MAX_WORDS
        (located if ok else verbose).append(r)
        if ok and short:
            both_ok.append(r)

    n = len(records)
    print(f"gold records                    {n}")
    print(f"  value locatable in the OCR    {len(located):4d}  {100*len(located)/n:5.1f}%")
    print(f"  value <= {MAX_WORDS} words             "
          f"{sum(1 for r in records if len(r['value'].split()) <= MAX_WORDS):4d}"
          f"  {100*sum(1 for r in records if len(r['value'].split()) <= MAX_WORDS)/n:5.1f}%")
    print(f"  both (usable as gold)         {len(both_ok):4d}  {100*len(both_ok)/n:5.1f}%")

    bad = [r for r in records if r not in both_ok]
    print(f"\nnot usable: {len(bad)}")
    by_section: dict[str, int] = {}
    for r in bad:
        by_section[r["section"]] = by_section.get(r["section"], 0) + 1
    for sec, c in sorted(by_section.items(), key=lambda x: -x[1]):
        total = sum(1 for r in records if r["section"] == sec)
        print(f"  {sec:24s} {c:4d} / {total}")

    if args.list:
        print()
        for r in bad:
            why = []
            if len(r["value"].split()) > MAX_WORDS:
                why.append("verbose")
            if _locate(r["value"], norm, starts, ends) is None:
                why.append("not in text")
            print(f"  [{','.join(why):18s}] {r['key']:28s} {r['value'][:70]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
