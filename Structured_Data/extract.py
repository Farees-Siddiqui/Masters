"""Open-schema key-value extraction from documents.

Stage-1 of the pipeline: given a document's text, ask Claude to emit every
key-value record it can find, grounded in verbatim evidence. This is the
"let the mess tell us what the schema is" step — run it on a handful of docs,
look at the keys that emerge, and that tells us what canonicalization and
metadata rules we actually need next.

Usage:
    python extract.py samples/resnet.txt                      # groq (free) by default
    python extract.py some_paper.pdf --backend ollama
    python extract.py some_paper.pdf --backend anthropic --out records.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import llm
import parsers
from schema import Record

DEFAULT_BACKEND = "groq"
DEFAULT_PARSER = "pypdf"

# The extraction contract. The examples are the ones from the project brief —
# they anchor the two families of records we care about: (1) bibliographic
# metadata whose keys are inferred from position, and (2) typed content
# mentions whose keys are inferred from meaning.
SYSTEM_PROMPT = """\
You extract structured key-value records from documents.

A *record* is a single {key, value} pair that captures a piece of the
document's essence. Two things make this task non-obvious:

1. The KEY is usually NOT stated in the document. You infer the semantic type
   of a value. If the text says "residual nets", the key is "architecture" —
   even though the word "architecture" never appears next to it.
2. The VALUE is almost always a span that appears in the text. Copy it.

Rules:
- Extract EVERY meaningful record you can find, not just the obvious ones.
  Datasets, methods/architectures, metrics and their results, tasks, venues,
  identifiers, dates, organizations, authors, claims — all are fair game.
- Keys must be short, lowercase, snake_case, and reusable across documents
  (e.g. "dataset", "architecture", "metric", "result", "date",
  "publishing_number", "category", "task"). Reuse the same key for the same
  kind of thing rather than inventing near-duplicates.
- Every record needs `evidence`: a short quote copied VERBATIM from the
  document (character-for-character) that the record is grounded in.
- One record per distinct fact. If a sentence mentions three datasets, emit
  three records.

Worked examples:

Text: "arXiv:1512.03385v1  [cs.CV]  10 Dec 2015"
Records:
  {key: "publishing_number", value: "1512.03385v1", evidence: "arXiv:1512.03385v1"}
  {key: "category", value: "cs.CV", evidence: "[cs.CV]"}
  {key: "date", value: "10 Dec 2015", evidence: "10 Dec 2015"}

Text: "...we evaluate residual nets ... 8x deeper than VGG nets [41] ... 3.57%
error on the ImageNet test set ... won the 1st place on the ILSVRC 2015
classification task ... 28% relative improvement on the COCO object detection
dataset."
Records:
  {key: "architecture", value: "residual nets", evidence: "we evaluate residual nets"}
  {key: "architecture", value: "VGG nets", evidence: "deeper than VGG nets"}
  {key: "dataset", value: "ImageNet", evidence: "on the ImageNet test set"}
  {key: "dataset", value: "COCO", evidence: "the COCO object detection"}
  {key: "metric", value: "3.57% error", evidence: "3.57% error on the ImageNet test set"}
  {key: "task", value: "object detection", evidence: "COCO object detection dataset"}
"""


def read_document(path: Path, parser: str = DEFAULT_PARSER) -> str:
    """Load text from a .txt/.md file or extract it from a PDF via `parser`."""
    return parsers.parse(path, parser=parser)


def chunk_text(text: str, max_chars: int = 6000) -> list[str]:
    """Split on blank lines, greedily packing paragraphs up to max_chars.

    Keeps whole paragraphs together so evidence spans never straddle a chunk
    boundary. Good enough for stage-1; a layout-aware splitter can come later.
    """
    paras = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > max_chars:
            chunks.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf)
    return chunks or [text]


def extract_chunk(chunk: str, backend: str, model: str | None) -> list[Record]:
    """One structured-output call: chunk text -> records."""
    result = llm.extract(
        SYSTEM_PROMPT, f"Document:\n\n{chunk}", backend=backend, model=model
    )
    return result.records


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _ground_key(s: str) -> str:
    """Collapse whitespace and hyphens so line-wrap artifacts don't defeat matching.

    Real PDF text splits words across lines ("im-\\nprovement", "im provement"),
    and models silently repair them in their quotes. Stripping all whitespace and
    hyphens before comparing lets grounding survive those artifacts.
    """
    return re.sub(r"[\s\-]+", "", s).lower()


def verify_grounding(record: Record, source: str) -> bool:
    """True if the record's evidence quote actually appears in the source."""
    if not record.evidence.strip():
        return False
    return _ground_key(record.evidence) in _ground_key(source)


def dedupe(records: list[Record]) -> list[Record]:
    """Collapse records with the same (key, value), keeping the most confident."""
    rank = {"high": 3, "medium": 2, "low": 1}
    best: dict[tuple[str, str], Record] = {}
    for r in records:
        k = (r.key.lower().strip(), _norm(r.value))
        cur = best.get(k)
        if cur is None or rank[r.confidence.value] > rank[cur.confidence.value]:
            best[k] = r
    return list(best.values())


def extract_document(
    path: Path,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    parser: str = DEFAULT_PARSER,
    dedupe_records: bool = True,
) -> dict:
    text = read_document(path, parser=parser)

    all_records: list[Record] = []
    for chunk in chunk_text(text):
        all_records.extend(extract_chunk(chunk, backend, model))

    records = dedupe(all_records) if dedupe_records else all_records

    out = []
    for r in records:
        grounded = verify_grounding(r, text)
        out.append(
            {
                "key": r.key,
                "value": r.value,
                "evidence": r.evidence,
                "confidence": r.confidence.value,
                "grounded": grounded,
            }
        )
    # Grounded records first, then by confidence.
    rank = {"high": 3, "medium": 2, "low": 1}
    out.sort(key=lambda d: (d["grounded"], rank[d["confidence"]]), reverse=True)
    return {"source": str(path), "record_count": len(out), "records": out}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="Document to extract (.txt, .md, or .pdf)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=["groq", "ollama", "anthropic"])
    ap.add_argument("--parser", default=DEFAULT_PARSER, choices=["pypdf", "mistral"])
    ap.add_argument("--model", default=None, help="Override the backend's default model")
    ap.add_argument("--no-dedupe", action="store_true", help="Keep every record, including duplicates")
    ap.add_argument("--out", type=Path, help="Write JSON here instead of stdout")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"No such file: {args.path}", file=sys.stderr)
        return 1

    result = extract_document(
        args.path,
        backend=args.backend,
        model=args.model,
        parser=args.parser,
        dedupe_records=not args.no_dedupe,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        n = result["record_count"]
        grounded = sum(r["grounded"] for r in result["records"])
        print(f"Wrote {n} records ({grounded} grounded) to {args.out}")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
