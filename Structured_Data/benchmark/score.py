"""Score predicted records against a gold answer key.

The shared dependency of both benchmark tracks: the synthetic one, where gold is
the seed a document was rendered from, and the real one, where gold is the
hand-read `Resnet/` answer key. Same scorer, same numbers, two sources of truth.

**Three nested levels, never one fused score.** A single number cannot tell
"found the fact but named it differently" apart from "missed the fact", and
those demand different fixes:

    value           the gold value was extracted at all
    value + key     ...and the key agrees (via the induced vocabulary)
    value + key + entity   ...and it is attached to the right subject

Each level is a strict subset of the one above, so the gaps between them read
directly as key-naming loss and entity-linkage loss.

**Key matching is induced, not authored.** Gold says `course`, the model says
`class_taught`. Nothing hard-codes that they agree. With `--induce`, gold keys
are pooled in with the predicted keys and `vocab.py` clusters the union in its
joint name+value space; two keys match iff they land in the same cluster. The
gold key is just another key in the pool. Without it, keys must match as
strings, which measures something stricter and less interesting.

**Matching is one-to-one.** Three passes — triple, then pair, then value — so a
gold record claims the best predicted record still available and no predicted
record is counted twice. Without that, one repeated value like `1st place`
would satisfy every gold record holding it.

Usage:
    python benchmark/score.py --gold benchmark/gold_resnet.json \\
                              --pred records.json records_tables.json \\
                              --induce --misses benchmark/misses.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------------------------------------------------------
# Normalization
#
# Every rule here is a deliberate decision about what counts as "the same
# value", and each one moves the score. They are printed with the results so
# the number is never quoted without them.
# --------------------------------------------------------------------------

RULES = [
    "casefold; collapse whitespace",
    "unicode dashes/quotes/x/± folded to ascii",
    "trailing sentence period dropped",
    "trailing % dropped (tables.py puts the unit in the key, the LLM in the value)",
    "citation brackets [41] dropped when something survives",
    "surrounding quotes dropped",
]

_FOLD = {
    "×": "x", "−": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "±": "+-", "≤": "<=", "≥": ">=", "∈": " in ",
    " ": " ",
}


def norm_value(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s or ""))
    for a, b in _FOLD.items():
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip().casefold()
    s = s.strip('"\'')
    stripped = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", s).strip()
    if stripped:                      # never let a bare "[41]" normalize to ""
        s = stripped
    s = re.sub(r"\.$", "", s).strip()
    if s.endswith("%") and s[:-1].strip():
        s = s[:-1].strip()
    return s


def norm_key(s: str) -> str:
    s = re.sub(r"[\s\-]+", "_", str(s or "").strip().casefold())
    return re.sub(r"_+", "_", s).strip("_")


def norm_entity(s: str) -> str:
    """Row labels get the same citation-bracket treatment as values.

    `tables.py` reads the row label verbatim, so Table 4's label arrives as
    `BN-inception [16]` while the gold set names the model `BN-inception`. The
    bracket is a reference marker, not part of the subject's name.
    """
    s = re.sub(r"\s+", " ", str(s or "")).strip().casefold()
    stripped = re.sub(r"\s*\[\d+(?:,\s*\d+)*\]", "", s).strip()
    return stripped or s


# A gold value of one or two words is an atomic fact an extractor can copy out.
# A ten-word value is a proposition the gold author wrote as a sentence, and no
# extractor will reproduce it verbatim, so string equality cannot score it.
ATOMIC_WORDS = 4


def is_atomic(value: str) -> bool:
    return len(str(value or "").split()) <= ATOMIC_WORDS


def is_table_sourced(src: str) -> bool:
    """Gold read out of a rendered table, vs read out of prose.

    The one axis the real paper already varies, and a preview of the synthetic
    benchmark's format axis: `T.3` is a table, `p.5` / `§3.4` / `A.` is prose.
    """
    return str(src or "").strip().upper().startswith("T.")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_records(path: Path) -> list[dict]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    recs = blob["records"] if isinstance(blob, dict) else blob
    out = []
    for r in recs:
        out.append({
            "entity": r.get("entity", ""),
            "key": r.get("key", ""),
            "raw_key": r.get("raw_key", "") or r.get("key", ""),
            "value": r.get("value", ""),
            "src": r.get("src", ""),
            "origin": path.name,
        })
    return out


# --------------------------------------------------------------------------
# Key equivalence
# --------------------------------------------------------------------------


class KeyMatcher:
    """Decides whether a gold key and a predicted key name the same thing."""

    def __init__(self, mode: str = "exact"):
        self.mode = mode
        self.canon: dict[str, str] = {}

    @classmethod
    def induced(cls, gold: list[dict], pred: list[dict], threshold: float) -> "KeyMatcher":
        """Pool both sides' keys and cluster the union.

        This is the whole argument in one function: the gold key is not special,
        it is just another emitted key with a value distribution. Nothing is
        hand-mapped, and a gold key that matches nothing stays itself.
        """
        self = cls(mode="induced")
        try:
            from vocab import Vocabulary
        except Exception as exc:
            print(f"[score] vocab.py unavailable ({exc}); falling back to exact keys",
                  file=sys.stderr)
            self.mode = "exact"
            return self

        key_values: dict[str, list[str]] = defaultdict(list)
        counts: Counter = Counter()
        for rec in list(gold) + list(pred):
            k = norm_key(rec["key"])
            key_values[k].append(str(rec["value"]))
            counts[k] += 1
        try:
            vocab = Vocabulary.induce(dict(key_values), counts, threshold=threshold)
            self.canon = vocab.assign(dict(key_values))
            self.backend = vocab.backend
            merged = {l: g for l, g in vocab.members.items() if len(g) > 1}
            print(f"[score] induced {len(vocab.labels)} clusters from "
                  f"{len(key_values)} keys ({vocab.backend}, t={threshold}); "
                  f"{len(merged)} merged")
        except Exception as exc:
            print(f"[score] induction failed ({exc}); falling back to exact keys",
                  file=sys.stderr)
            self.mode = "exact"
        return self

    def label(self, key: str) -> str:
        k = norm_key(key)
        return self.canon.get(k, k)

    def match(self, gold_key: str, pred: dict) -> bool:
        g = self.label(gold_key)
        return g == self.label(pred["key"]) or g == self.label(pred["raw_key"])


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

VALUE, KEY, ENTITY = 1, 2, 3


def score(gold: list[dict], pred: list[dict], km: KeyMatcher) -> dict:
    """Greedy one-to-one assignment, best level first."""
    by_value: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(pred):
        by_value[norm_value(p["value"])].append(i)

    used: set[int] = set()
    level: dict[int, int] = {}
    partner: dict[int, int] = {}

    def entity_ok(g: dict, p: dict) -> bool:
        # Gold with no entity states no linkage, so there is nothing to get
        # wrong -- it cannot be charged as an entity failure.
        ge = norm_entity(g["entity"])
        return not ge or ge == norm_entity(p["entity"])

    passes = [
        (ENTITY, lambda g, p: km.match(g["key"], p) and entity_ok(g, p)),
        (KEY,    lambda g, p: km.match(g["key"], p)),
        (VALUE,  lambda g, p: True),
    ]
    for lvl, ok in passes:
        for gi, g in enumerate(gold):
            if gi in level:
                continue
            for pi in by_value.get(norm_value(g["value"]), ()):
                if pi in used or not ok(g, pred[pi]):
                    continue
                used.add(pi)
                level[gi] = lvl
                partner[gi] = pi
                break

    return {"level": level, "partner": partner, "used": used}


def summarize(gold: list[dict], pred: list[dict], res: dict) -> dict:
    lvl = res["level"]
    n = len(gold)
    at = lambda t: sum(1 for v in lvl.values() if v >= t)

    def subset(pick) -> tuple[int, int, int, int]:
        idx = [i for i, g in enumerate(gold) if pick(g)]
        return (len(idx),
                sum(1 for i in idx if lvl.get(i, 0) >= VALUE),
                sum(1 for i in idx if lvl.get(i, 0) >= KEY),
                sum(1 for i in idx if lvl.get(i, 0) >= ENTITY))

    return {
        "gold": n,
        "pred": len(pred),
        "value": at(VALUE),
        "key": at(KEY),
        "entity": at(ENTITY),
        "pred_matched": len(res["used"]),
        "table": subset(lambda g: is_table_sourced(g["src"])),
        "prose": subset(lambda g: not is_table_sourced(g["src"])),
        "atomic": subset(lambda g: is_atomic(g["value"])),
        "propositional": subset(lambda g: not is_atomic(g["value"])),
    }


def pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:5.1f}%" if b else "    --"


def report(name: str, s: dict) -> None:
    print(f"\n{name}")
    print(f"  gold {s['gold']}   predicted {s['pred']}")
    print(f"  {'level':<24}{'hits':>7}{'recall':>9}")
    print(f"  {'-' * 40}")
    for label, k in (("value", "value"), ("+ key", "key"), ("+ entity", "entity")):
        print(f"  {label:<24}{s[k]:>7}{pct(s[k], s['gold']):>9}")
    print(f"  {'-' * 40}")
    print(f"  {'predicted matched':<24}{s['pred_matched']:>7}{pct(s['pred_matched'], s['pred']):>9}")
    for title, groups in (
        ("by source surface", ("table", "prose")),
        (f"by gold granularity (atomic = value <= {ATOMIC_WORDS} words)",
         ("atomic", "propositional")),
    ):
        print(f"\n  {title}")
        print(f"  {'':<18}{'n':>7}{'value':>10}{'+key':>10}{'+entity':>10}")
        for label in groups:
            n, v, k, e = s[label]
            print(f"  {label:<18}{n:>7}{pct(v, n):>10}{pct(k, n):>10}{pct(e, n):>10}")


def write_misses(path: Path, gold: list[dict], pred: list[dict], res: dict) -> None:
    lvl, partner = res["level"], res["partner"]
    lines = ["# Scoring detail", "",
             "Every gold record the scorer could not fully match, so the "
             "disagreements can be read by hand against the PDF.", ""]

    for title, keep in (
        ("Value never found", lambda i: i not in lvl),
        ("Value found, key disagreed", lambda i: lvl.get(i) == VALUE),
        ("Value + key found, entity disagreed", lambda i: lvl.get(i) == KEY),
    ):
        rows = [i for i in range(len(gold)) if keep(i)]
        lines += [f"## {title} ({len(rows)})", ""]
        if not rows:
            lines += ["_none_", ""]
            continue
        lines += ["| src | gold entity | gold key | gold value | matched to |",
                  "|---|---|---|---|---|"]
        for i in rows[:400]:
            g = gold[i]
            p = pred[partner[i]] if i in partner else None
            got = (f"`{p['key']}` / {p['entity'] or '—'} ({p['origin']})") if p else "—"
            cell = lambda s: str(s).replace("|", "\\|")[:80]
            lines.append(f"| {cell(g['src'])} | {cell(g['entity']) or '—'} | "
                         f"`{cell(g['key'])}` | {cell(g['value'])} | {cell(got)} |")
        if len(rows) > 400:
            lines.append(f"| … | | | _{len(rows) - 400} more_ | |")
        lines.append("")

    unmatched = [p for i, p in enumerate(pred) if i not in res["used"]]
    lines += [f"## Predicted records matching no gold record ({len(unmatched)})", "",
              "Not necessarily wrong — the gold set is one reading of the paper, "
              "and an extra correct fact lands here too.", "",
              "| origin | entity | key | value |", "|---|---|---|---|"]
    for p in unmatched[:400]:
        cell = lambda s: str(s).replace("|", "\\|")[:80]
        lines.append(f"| {p['origin']} | {cell(p['entity']) or '—'} | "
                     f"`{cell(p['key'])}` | {cell(p['value'])} |")
    if len(unmatched) > 400:
        lines.append(f"| … | | | _{len(unmatched) - 400} more_ |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ndetail -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--induce", action="store_true",
                    help="match keys by clustering gold+predicted keys together")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--misses", type=Path, help="write a readable disagreement report")
    args = ap.parse_args()

    gold = load_records(args.gold)
    preds = {p.name: load_records(p) for p in args.pred}
    union = [r for rs in preds.values() for r in rs]

    km = (KeyMatcher.induced(gold, union, args.threshold)
          if args.induce else KeyMatcher("exact"))

    print(f"\ngold: {args.gold}  ({len(gold)} records)")
    print(f"key matching: {km.mode}")
    print("value normalization:")
    for r in RULES:
        print(f"  - {r}")

    if len(preds) > 1:
        for name, recs in preds.items():
            report(name, summarize(gold, recs, score(gold, recs, km)))

    res = score(gold, union, km)
    report(" + ".join(preds) if len(preds) > 1 else next(iter(preds)),
           summarize(gold, union, res))

    if args.misses:
        args.misses.parent.mkdir(parents=True, exist_ok=True)
        write_misses(args.misses, gold, union, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
