"""Open-schema key-value extraction from documents.

Given a document's text, ask a model to emit every key-value record it can
find, grounded in verbatim evidence, then resolve that evidence to real
character offsets in the source.

The model stays open-schema — it is never handed a fixed list of allowed keys,
because the extractor has to work on document types nobody has seen. What it
*is* optionally handed is a vocabulary **induced from previous runs**
(`vocab.py`), as a preference rather than a constraint: reuse these keys where
they fit, invent one where they don't.

Usage:
    python extract.py samples/resnet.txt                       # local ollama by default
    python extract.py some_paper.pdf --model llama3.3:70b
    python extract.py paper.pdf --vocab vocab.json --out records.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import llm
import parsers
from schema import Confidence, GroundedRecord, Record, Span

DEFAULT_BACKEND = "ollama"
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

# Appended only when a vocabulary has been induced. Deliberately a preference,
# not a constraint — a hard allow-list would break on unseen document types.
_VOCAB_HINT = """
Keys seen in this corpus so far, listed to keep naming consistent:
{keys}

Prefer one of these when it genuinely fits the value. If none fits, invent a
new key rather than forcing a bad match — new document types are expected to
introduce new keys.
"""


def read_document(path: Path, parser: str = DEFAULT_PARSER) -> str:
    """Load text from a .txt/.md file or extract it from a PDF via `parser`."""
    return parsers.parse(path, parser=parser)


def chunk_text(text: str, max_chars: int = 6000) -> list[tuple[int, str]]:
    """Split on blank lines, greedily packing paragraphs up to max_chars.

    Returns (offset, chunk) so a span found inside a chunk maps straight back
    to an absolute position in the document. Keeps whole paragraphs together so
    evidence spans never straddle a chunk boundary.
    """
    chunks: list[tuple[int, str]] = []
    buf, buf_start, cursor = "", 0, 0
    for para in re.split(r"(\n\s*\n)", text):
        if not para:
            continue
        if re.fullmatch(r"\n\s*\n", para):
            buf += para
            cursor += len(para)
            continue
        if buf.strip() and len(buf) + len(para) > max_chars:
            chunks.append((buf_start, buf))
            buf, buf_start = para, cursor
        else:
            if not buf.strip():
                buf_start = cursor - (len(buf))
            buf += para
        cursor += len(para)
    if buf.strip():
        chunks.append((buf_start, buf))
    return chunks or [(0, text)]


def extract_chunk(
    chunk: str, backend: str, model: str | None, vocab_hint: str = ""
) -> list[Record]:
    """One structured-output call: chunk text -> records."""
    result = llm.extract(
        SYSTEM_PROMPT + vocab_hint,
        f"Document:\n\n{chunk}",
        backend=backend,
        model=model,
    )
    return result.records


# --------------------------------------------------------------------------
# Grounding: resolve a quoted span to real character offsets.
# --------------------------------------------------------------------------


# Markup that sits *between* a value's characters in OCR output and would
# otherwise defeat matching: markdown emphasis, LaTeX math delimiters, quotes,
# the braces of `{a, b}@host.com` email notation, table pipes, and the brackets
# of inline citations (`VGG nets [41] are`) — all structural, never part of a
# value, so dropping them from the matching view only ever helps.
_SKIP_CHARS = set(" \t\r\n-*_`\"'“”‘’{}()[]|~^$\\")

# LaTeX escapes the OCR emits inside math spans. Resolved to the character they
# represent so `$112 \times 112$` matches a quoted `112 × 112`.
_LATEX = {
    r"\times": "×", r"\%": "%", r"\$": "$", r"\&": "&", r"\#": "#",
    r"\_": "_", r"\cdot": "·", r"\pm": "±", r"\leq": "≤", r"\geq": "≥",
    r"\le": "≤", r"\ge": "≥", r"\ldots": "…", r"\dots": "…",
}
_LATEX_MAX = max(len(k) for k in _LATEX)


def _norm_index(text: str) -> tuple[str, list[int], list[int]]:
    """Build a markup-free view of `text` plus maps back to real offsets.

    Real OCR text splits words across lines ("im-\\nprovement"), bolds numbers
    (`**3.57%**`), quotes phrases, and wraps math in LaTeX — and models silently
    repair all of that when they quote. Comparing in a stripped view survives
    it; carrying the position lists means we still report exact offsets into the
    *original* text, so grounding never degrades into a fuzzy match.

    Returns (normalized, starts, ends). A normalized character can stand for a
    run of source characters (`\\times` -> `×`), so starts/ends are tracked
    separately rather than assuming a one-to-one mapping.
    """
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\":
            hit = next(
                (k for k in (text[i : i + L] for L in range(_LATEX_MAX, 1, -1)) if k in _LATEX),
                None,
            )
            if hit:
                chars.append(_LATEX[hit])
                starts.append(i)
                ends.append(i + len(hit))
                i += len(hit)
                continue
        if text[i] in _SKIP_CHARS:
            i += 1
            continue
        chars.append(text[i].lower())
        starts.append(i)
        ends.append(i + 1)
        i += 1
    return "".join(chars), starts, ends


def _locate(
    needle: str,
    norm: str,
    starts: list[int],
    ends: list[int],
    lo: int = 0,
    hi: int | None = None,
) -> tuple[int, int, int, int] | None:
    """Find `needle` in the normalized view. Returns (start, end, jlo, jhi)."""
    nrm, _, _ = _norm_index(needle)
    if not nrm:
        return None
    j = norm.find(nrm, lo, hi if hi is not None else len(norm))
    if j < 0:
        return None
    return starts[j], ends[j + len(nrm) - 1], j, j + len(nrm)


def resolve(
    record: Record, text: str, norm: str, starts: list[int], ends: list[int]
) -> tuple[Span, str] | None:
    """Locate a record's value in the source, preferring inside its evidence.

    Returns the span and *how* it was found, because those are not equally
    strong claims and collapsing them into one boolean inflates the grounding
    rate. Only `value_in_evidence` and `value_in_document` anchor the value
    itself; `evidence_only` means the value was never found anywhere and the
    span points at the sentence the model was looking at instead. Report them
    separately — `evidence_only` is where invented values hide.
    """
    ev = _locate(record.evidence, norm, starts, ends)
    if ev is not None:
        val = _locate(record.value, norm, starts, ends, lo=ev[2], hi=ev[3])
        if val is not None:
            return Span(start=val[0], end=val[1], text=text[val[0] : val[1]]), "value_in_evidence"

    # Value not inside its own evidence: the model abstracted rather than
    # copied ("residual nets" from "a residual learning framework").
    val = _locate(record.value, norm, starts, ends)
    if val is not None:
        return Span(start=val[0], end=val[1], text=text[val[0] : val[1]]), "value_in_document"
    if ev is not None:
        return Span(start=ev[0], end=ev[1], text=text[ev[0] : ev[1]]), "evidence_only"
    return None


# --------------------------------------------------------------------------
# Dedupe: one record per distinct fact, every mention kept as a span.
# --------------------------------------------------------------------------

_ARTICLES = re.compile(r"^(the|a|an)\s+")

# Strength order for grounding claims; see `resolve`.
_GROUND_RANK = {
    "value_in_evidence": 3,
    "value_in_document": 2,
    "evidence_only": 1,
    "none": 0,
}


def _canon_value(s: str) -> str:
    v = re.sub(r"[^\w\s%.+-]", " ", s.lower())
    v = re.sub(r"\s+", " ", v).strip()
    return _ARTICLES.sub("", v)


def collapse(records: list[GroundedRecord]) -> list[GroundedRecord]:
    """Collapse to one record per (key, value), merging their spans.

    This is what makes `record_count` stable: it counts distinct facts, so a
    model that mentions ImageNet nine times and a model that mentions it twice
    agree on the count and differ only in `mentions`.
    """
    rank = {Confidence.high: 3, Confidence.medium: 2, Confidence.low: 1}
    best: dict[tuple[str, str], GroundedRecord] = {}
    for r in records:
        k = (r.key, _canon_value(r.value))
        cur = best.get(k)
        if cur is None:
            best[k] = r.model_copy(deep=True)
            continue
        seen = {(s.start, s.end) for s in cur.spans}
        cur.spans.extend(s for s in r.spans if (s.start, s.end) not in seen)
        # Keep the strongest grounding claim across the merged mentions: one
        # mention locating the value is enough to say the value exists.
        if _GROUND_RANK[r.grounded_by] > _GROUND_RANK[cur.grounded_by]:
            cur.grounded_by = r.grounded_by
        if rank[r.confidence] > rank[cur.confidence]:
            cur.confidence = r.confidence
            cur.value = r.value
            cur.evidence = r.evidence
    for r in best.values():
        r.spans.sort(key=lambda s: s.start)
    return list(best.values())


def extract_document(
    path: Path,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    parser: str = DEFAULT_PARSER,
    dedupe_records: bool = True,
    vocab_path: Path | None = None,
) -> dict:
    text = read_document(path, parser=parser)

    vocabulary = None
    hint = ""
    if vocab_path and vocab_path.exists():
        from vocab import Vocabulary

        vocabulary = Vocabulary.load(vocab_path)
        hint = _VOCAB_HINT.format(keys=", ".join(sorted(vocabulary.labels)))

    raw: list[Record] = []
    for _, chunk in chunk_text(text):
        raw.extend(extract_chunk(chunk, backend, model, hint))

    norm, starts, ends = _norm_index(text)

    # Canonicalize in one batch so every key is placed using the same joint
    # name+value representation the vocabulary was induced with.
    raw_keys = [r.key.strip().lower().replace(" ", "_").replace("-", "_") for r in raw]
    mapping: dict[str, str] = {}
    if vocabulary:
        key_values: dict[str, list[str]] = {}
        for rk, r in zip(raw_keys, raw):
            key_values.setdefault(rk, []).append(r.value)
        mapping = vocabulary.assign(key_values)

    grounded: list[GroundedRecord] = []
    for raw_key, r in zip(raw_keys, raw):
        key = mapping.get(raw_key, raw_key)
        found = resolve(r, text, norm, starts, ends)
        span, how = found if found else (None, "none")
        grounded.append(
            GroundedRecord(
                key=key,
                raw_key=raw_key,
                value=r.value,
                confidence=r.confidence,
                spans=[span] if span else [],
                evidence=r.evidence,
                grounded_by=how,
            )
        )

    records = collapse(grounded) if dedupe_records else grounded

    rank = {Confidence.high: 3, Confidence.medium: 2, Confidence.low: 1}
    records.sort(key=lambda r: (r.grounded, rank[r.confidence], r.mentions), reverse=True)

    out = [
        {
            "key": r.key,
            "raw_key": r.raw_key,
            "value": r.value,
            "evidence": r.evidence,
            "confidence": r.confidence.value,
            "grounded": r.grounded,
            "value_grounded": r.value_grounded,
            "grounded_by": r.grounded_by,
            "mentions": r.mentions,
            "spans": [s.model_dump() for s in r.spans],
        }
        for r in records
    ]
    by = Counter(d["grounded_by"] for d in out)
    return {
        "source": str(path),
        "record_count": len(out),
        # The strict number: the value itself was located in the source.
        "value_grounded_count": sum(d["value_grounded"] for d in out),
        # The loose number, kept for continuity. Includes `evidence_only`,
        # where only the surrounding sentence was found — do not quote this
        # one as a grounding rate.
        "grounded_count": sum(d["grounded"] for d in out),
        "grounded_by": dict(by),
        "distinct_keys": len({d["key"] for d in out}),
        "distinct_raw_keys": len({d["raw_key"] for d in out}),
        "vocab": str(vocab_path) if vocabulary else None,
        "records": out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="Document to extract (.txt, .md, or .pdf)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=["ollama", "anthropic"])
    ap.add_argument("--parser", default=DEFAULT_PARSER, choices=["pypdf", "mistral"])
    ap.add_argument("--model", default=None, help="Override the backend's default model")
    ap.add_argument("--vocab", type=Path, default=None, help="Induced vocabulary (vocab.json)")
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
        vocab_path=args.vocab,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        args.out.write_text(payload, encoding="utf-8")
        print(
            f"Wrote {result['record_count']} records to {args.out}\n"
            f"  value-grounded {result['value_grounded_count']} "
            f"({result['value_grounded_count'] / max(result['record_count'], 1):.1%})"
            f"  evidence-only {result['grounded_by'].get('evidence_only', 0)}"
            f"  ungrounded {result['grounded_by'].get('none', 0)}\n"
            f"  keys {result['distinct_keys']} from {result['distinct_raw_keys']} raw"
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
