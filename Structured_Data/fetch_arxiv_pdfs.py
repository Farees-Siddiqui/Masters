#!/usr/bin/env python3
"""Download the 10 survey papers as published arXiv PDFs.

Reads only the arxiv_id fields from samples/arxiv_latex/manifest.json. No .tex
file is opened here or anywhere else in the benchmark -- the manifest is
metadata, and every downstream stage works from the rendered PDF alone.
"""
import json
import pathlib
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "samples" / "arxiv_latex" / "manifest.json"
OUT = ROOT / "arxiv_papers"

# arXiv asks automated clients to identify themselves and to space out requests.
UA = "Mozilla/5.0 (compatible; thesis-ocr-survey/1.0; +farees.siddiqui@ontariotechu.net)"
DELAY_SEC = 3.0


def main() -> int:
    OUT.mkdir(exist_ok=True)
    papers = {k: v["arxiv_id"] for k, v in json.loads(MANIFEST.read_text()).items()}

    failed = []
    for i, (name, arxiv_id) in enumerate(sorted(papers.items())):
        dest = OUT / f"{name}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[skip] {name:12s} {arxiv_id:14s} already present "
                  f"({dest.stat().st_size:,} B)")
            continue
        if i:
            time.sleep(DELAY_SEC)
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read()
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {name:12s} {arxiv_id:14s} {exc}")
            failed.append(name)
            continue
        if not blob.startswith(b"%PDF"):
            print(f"[FAIL] {name:12s} {arxiv_id:14s} not a PDF ({len(blob)} B)")
            failed.append(name)
            continue
        dest.write_bytes(blob)
        print(f"[ok]   {name:12s} {arxiv_id:14s} {len(blob):,} B")

    print(f"\n{len(papers) - len(failed)}/{len(papers)} PDFs in {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
