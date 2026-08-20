"""Lift the hand-made ResNet answer key out of LaTeX into scorable JSON.

`Resnet/sec_*.tex` holds 868 key-value records read off `samples/resnet.pdf` by
hand. They are laid out for a human — coloured columns, math, escaped specials —
and no scorer can read that. This module parses them back into the shape the
rest of the pipeline uses (`entity`, `key`, `value`), plus the source locator
that says where in the paper each record came from.

The rows are rigidly formatted, so this is parsing and not inference:

    \\K{key} & value & \\Sd{src}\\\\                    (two-column table)
    \\E{entity} & \\K{key} & value & \\Sd{src}\\\\      (three-column table)

Two details are load-bearing:

- Cells are split at brace depth zero. A raw `&` occurs inside `\\url{...}` in
  the metadata section (a VOC leaderboard query string), and splitting on every
  `&` would tear that row in half.
- `src` is kept because it is the one axis the real document already varies:
  `T.3` means the fact was read out of a rendered table, `p.5` or `§3.4` means
  it was read out of prose. That is a preview of the synthetic benchmark's
  format axis, available on real data today.

Usage:
    python benchmark/tex_to_gold.py Resnet --out benchmark/gold_resnet.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# The order sections are \input in resnet_kv.tex, so record ids follow the
# document rather than the filesystem's idea of alphabetical.
SECTION_ORDER = [
    "sec_metadata",
    "sec_contributions",
    "sec_method",
    "sec_architecture",
    "sec_training",
    "sec_results_imagenet",
    "sec_results_cifar",
    "sec_detection",
    "sec_localization",
    "sec_references",
]

# --------------------------------------------------------------------------
# LaTeX -> plain text
# --------------------------------------------------------------------------

# Applied in order. Accents and wrappers first, then symbols, then the escaped
# specials — `\_` must survive long enough not to be mistaken for a macro name.
_SUBS: list[tuple[str, str]] = [
    (r'\\"\{([aeiou])\}', lambda m: {"a":"\u00e4","e":"\u00eb","i":"\u00ef",
                                     "o":"\u00f6","u":"\u00fc"}[m.group(1)]),
    (r"\\'\{([aeiou])\}", lambda m: {"a":"\u00e1","e":"\u00e9","i":"\u00ed",
                                     "o":"\u00f3","u":"\u00fa"}[m.group(1)]),
    (r"\\mathcal\{([A-Za-z])\}", r"\1"),
    (r"\\mathbf\{([A-Za-z])\}", r"\1"),
    (r"\\(?:emph|textbf|textit|texttt|text|url)\{", "{"),
    # Control *words* end at the first non-letter and swallow one following
    # space, exactly as TeX does: `3\times3` and `\times 2` are both `x`. A
    # trailing \b cannot express this — there is no word boundary between the
    # `s` of `times` and the `3` that follows it.
    (r"\\times(?![A-Za-z]) ?", "\u00d7"),
    (r"\\pm(?![A-Za-z]) ?", "\u00b1"),
    (r"\\in(?![A-Za-z]) ?", "\u2208"),
    (r"\\sigma(?![A-Za-z]) ?", "\u03c3"),
    (r"\\dagger(?![A-Za-z]) ?", "\u2020"),
    (r"\\leq(?![A-Za-z]) ?", "\u2264"),
    (r"\\ldots(?![A-Za-z]) ?", "..."),
    # The OCR writes 10⁴ with a real superscript character.
    (r"\\textsuperscript\{4\}", "\u2074"),
    (r"\\S(?=\d)", "\u00a7"),
    (r"\\%", "%"),
    (r"\\&", "&"),
    (r"\\_", "_"),
    (r"\\#", "#"),
    (r"\\\$", "$"),
    (r"\\\{", "\x01"),   # parked: real braces, not grouping
    (r"\\\}", "\x02"),
    (r"\\,", ""),        # thin space
    (r"\\;", " "),
    (r"\\ ", " "),       # `W.\ L.\ Briggs`
    (r"---", "\u2014"),
    (r"--", "\u2013"),
    (r"``", '"'),
    (r"''", '"'),
    (r"\^\{([^{}]*)\}", r"^\1"),
    (r"_\{([^{}]*)\}", r"_\1"),
]


def detex(s: str) -> str:
    """Reduce a LaTeX cell to the plain string a scorer can compare."""
    for pat, rep in _SUBS:
        s = re.sub(pat, rep, s)
    s = s.replace("$", "").replace("~", " ")
    # Whatever grouping braces remain carried no meaning of their own.
    s = s.replace("{", "").replace("}", "")
    s = s.replace("\x01", "{").replace("\x02", "}")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------
# Row parsing
# --------------------------------------------------------------------------


def split_cells(row: str) -> list[str]:
    """Split a table row on `&` at brace depth zero.

    `\\url{...?challengeid=11&compid=4}` puts a genuine ampersand inside braces;
    a naive split on `&` would produce a five-cell row and drop the record.
    """
    cells, buf, depth = [], [], 0
    i = 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row):     # escaped char: pass both through
            buf.append(row[i : i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "&" and depth == 0:
            cells.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    cells.append("".join(buf))
    return [c.strip() for c in cells]


def unwrap(cell: str, macro: str) -> str | None:
    """Return the argument of `\\macro{...}`, brace-balanced, or None."""
    prefix = "\\" + macro + "{"
    if not cell.startswith(prefix):
        return None
    depth, out = 1, []
    i = len(prefix)
    while i < len(cell) and depth:
        ch = cell[i]
        if ch == "\\" and i + 1 < len(cell):
            out.append(cell[i : i + 2])
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if not depth:
                break
        out.append(ch)
        i += 1
    return "".join(out)


BEGIN = re.compile(r"\\begin\{(e?kvtable)\}\{(.*)\}\{([^{}]*)\}\s*$")


def parse_section(path: Path) -> list[dict]:
    """Every record row in one section file, tagged with its table caption."""
    records: list[dict] = []
    caption = label = ""
    arity = 0

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        m = BEGIN.match(line)
        if m:
            kind, caption, label = m.group(1), detex(m.group(2)), m.group(3)
            arity = 4 if kind == "ekvtable" else 3
            continue
        if line.startswith("\\end{") or not line.endswith("\\\\"):
            continue
        if not (line.startswith("\\E{") or line.startswith("\\K{")):
            continue  # \addlinespace, prose, header furniture

        cells = split_cells(line[:-2])
        if len(cells) != arity:
            raise ValueError(f"{path.name}: expected {arity} cells, got {len(cells)}\n  {line}")

        if arity == 4:
            entity, key, value, src = cells
            entity = unwrap(entity, "E") or entity
        else:
            entity, (key, value, src) = "", cells

        key = unwrap(key, "K")
        src = unwrap(src, "Sd") or src
        if key is None:
            raise ValueError(f"{path.name}: row without \\K{{...}}\n  {line}")

        records.append({
            "entity": detex(entity),
            "key": detex(key),
            "value": detex(value),
            "src": detex(src),
            "table": caption,
            "label": label,
            "section": path.stem,
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir", type=Path, help="folder holding sec_*.tex (e.g. Resnet)")
    ap.add_argument("--source", default="samples/resnet.pdf",
                    help="document the records were read from")
    ap.add_argument("--out", type=Path, default=Path("benchmark/gold_resnet.json"))
    args = ap.parse_args()

    records: list[dict] = []
    for name in SECTION_ORDER:
        path = args.dir / f"{name}.tex"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        got = parse_section(path)
        print(f"  {name:24s} {len(got):4d}")
        records.extend(got)

    for i, rec in enumerate(records):
        rec["id"] = i

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": args.source,
        "origin": f"{args.dir}/sec_*.tex (hand-read from the PDF)",
        "record_count": len(records),
        "distinct_keys": len({r["key"] for r in records}),
        "distinct_entities": len({r["entity"] for r in records if r["entity"]}),
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(records)} records, {len({r['key'] for r in records})} keys, "
          f"{len({r['entity'] for r in records if r['entity']})} entities -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
