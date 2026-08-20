"""Key-value records parsed straight out of markdown tables.

A table is already a key-value structure and nobody has to infer it:

    |  model        | top-1 err. | top-5 err. |
    | ---           | ---        | ---        |
    |  VGG-16 [41]  | 28.07      | 9.33       |

    -> {entity: "VGG-16 [41]", key: "top_1_err", value: "28.07"}
    -> {entity: "VGG-16 [41]", key: "top_5_err", value: "9.33"}
    -> {entity: "VGG-16 [41]", key: "model",     value: "VGG-16 [41]"}

The column header *is* the key, the cell *is* the value, the row label *is* the
entity. This is parsing, not inference: the output is a function of the input,
identical on every run, with exact character offsets. No model is involved and
no key is hard-coded — every key is read off the document's own header row.

This matters beyond determinism. Asked to extract from these rows, an LLM reads
`28.07`, correctly infers from the header that it is a percentage, and emits
`28.07%` — a value that appears nowhere in the source and so cannot be grounded
to a span. The header it inferred that from is right there in the text. Parsing
it keeps both the value and its unit without anything being invented.

Usage:
    python tables.py samples/resnet.mistral.md --out records_tables.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from schema import Confidence, GroundedRecord, Span

# A markdown separator row: pipes, dashes, colons, spaces, nothing else.
_SEPARATOR = re.compile(r"^\s*\|[\s\-:|]+\|?\s*$")

# Cells that carry no value. `-` and `n/a` are the document saying "not measured".
_EMPTY_CELLS = {"", "-", "--", "---", "—", "n/a", "na", "none"}


def _line_offsets(text: str) -> list[tuple[str, int]]:
    """Every line with its absolute start offset in `text`."""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((line, pos))
        pos += len(line) + 1
    return out


def _cells(line: str, line_start: int) -> list[tuple[str, int, int] | None]:
    """Split a `| a | b |` row into (text, start, end), None for empty cells.

    Offsets point at the *trimmed* content so a span never includes the padding
    or the pipes, and `text[start:end]` is exactly the value.
    """
    bars = [i for i, ch in enumerate(line) if ch == "|"]
    cells: list[tuple[str, int, int] | None] = []
    for a, b in zip(bars, bars[1:]):
        raw = line[a + 1 : b]
        stripped = raw.strip()
        if stripped.lower() in _EMPTY_CELLS:
            cells.append(None)
            continue
        lead = len(raw) - len(raw.lstrip())
        start = line_start + a + 1 + lead
        cells.append((stripped, start, start + len(stripped)))
    return cells


def _key_from_header(header: str) -> str:
    """`top-1 err.` -> `top_1_err`. Derived from the document, never authored."""
    k = header.strip().lower()
    k = re.sub(r"\\[a-z]+", " ", k)          # strip LaTeX commands
    k = re.sub(r"[^\w\s]", " ", k)           # punctuation -> space
    k = re.sub(r"\s+", "_", k.strip())
    return k.strip("_")


def find_tables(text: str) -> list[tuple[list[str | None], list[list]]]:
    """Locate markdown tables. Returns (header_cells, data_rows) per table.

    A table is a separator row plus the header line above it and the contiguous
    pipe-rows below. That is the whole grammar; anything not matching it is left
    to the LLM stage rather than guessed at.
    """
    lines = _line_offsets(text)
    tables = []
    i = 0
    while i < len(lines):
        line, _ = lines[i]
        if not _SEPARATOR.match(line) or i == 0:
            i += 1
            continue
        head_line, head_start = lines[i - 1]
        if not head_line.strip().startswith("|"):
            i += 1
            continue
        header = [c[0] if c else None for c in _cells(head_line, head_start)]

        rows = []
        j = i + 1
        while j < len(lines) and lines[j][0].strip().startswith("|"):
            row_line, row_start = lines[j]
            if not _SEPARATOR.match(row_line):
                rows.append(_cells(row_line, row_start))
            j += 1
        if rows:
            tables.append((header, rows))
        i = j
    return tables


def extract(text: str) -> list[GroundedRecord]:
    """Every table cell as a grounded record. Deterministic, no model."""
    records: list[GroundedRecord] = []
    for header, rows in find_tables(text):
        # The row label is the first column that actually carries labels. Many
        # tables leave its header blank (`|  | plain | ResNet |`), so identify
        # it by position rather than by name.
        for row in rows:
            if not row or row[0] is None:
                entity = ""
            else:
                entity = row[0][0]

            for col, cell in enumerate(row):
                if cell is None or col >= len(header):
                    continue
                head = header[col]
                if not head:
                    # Unlabelled column: with no header there is no key to read
                    # off the document, and inventing one is exactly what this
                    # module exists to avoid.
                    continue
                key = _key_from_header(head)
                if not key:
                    continue
                value, start, end = cell
                records.append(
                    GroundedRecord(
                        key=key,
                        raw_key=key,
                        value=value,
                        confidence=Confidence.high,
                        spans=[Span(start=start, end=end, text=text[start:end])],
                        evidence=value,
                        entity=entity if entity != value else "",
                        grounded_by="value_in_evidence",
                        extractor="table",
                    )
                )
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="Markdown/text document")
    ap.add_argument("--out", type=Path, help="Write JSON here instead of stdout")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"No such file: {args.path}", file=sys.stderr)
        return 1

    text = args.path.read_text(encoding="utf-8")
    records = extract(text)

    bad = [r for r in records for s in r.spans if text[s.start : s.end] != s.text]
    payload = {
        "source": str(args.path),
        "extractor": "table",
        "table_count": len(find_tables(text)),
        "record_count": len(records),
        "distinct_keys": len({r.key for r in records}),
        "distinct_entities": len({r.entity for r in records if r.entity}),
        "span_mismatches": len(bad),
        "records": [
            {
                "key": r.key,
                "entity": r.entity,
                "value": r.value,
                "extractor": r.extractor,
                "spans": [s.model_dump() for s in r.spans],
            }
            for r in records
        ],
    }
    out = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(out, encoding="utf-8")
        print(f"Wrote {len(records)} records from {payload['table_count']} tables "
              f"({payload['distinct_keys']} keys, {payload['distinct_entities']} entities, "
              f"{len(bad)} span mismatches) to {args.out}")
        print("top keys:", Counter(r.key for r in records).most_common(8))
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
