"""Gold construction: ``node -> {box keys} | ABSENT``, tiered by provenance.

Three sources, strongest first:

1. **human** — verdicts from ``layout_output/<doc>/annotations.json`` written
   by the /annotate tool. An ``excluded:<node>`` verdict of ``boxes`` asserts
   the node's boxes outright; ``absent`` asserts the node appears on no page.
   A ``null:<box>`` verdict of ``node`` adds that box to a node's gold;
   ``furniture`` confirms the box belongs to nobody.
2. **audited** — oracle placements a human accepted in the audit queue.
3. **oracle** — oracle placements nobody has reviewed yet. Included by
   default so the benchmark runs before annotation is complete, but every
   report states the tier composition — a result resting mostly on tier 3 is
   labelled as such rather than passed off as verified.

Audit *rejects* remove a node from gold entirely (its true location is unknown
until an excluded-queue-style annotation supplies one).

The oracle's token spans are converted to boxes by **majority ownership**: a
box belongs to a node when that node owns more than half of the box's
oracle-owned tokens. Majority (not plurality) keeps the mapping conservative —
a box genuinely straddling two nodes joins neither node's gold rather than
being awarded to whichever side has one token more.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from alignment.naive_aligner import _iter_segments
from benchmark.oracle import (
    _norm,
    boxes_to_tokens,
    build_stream,
    extract_tokens,
    gold_token_owner,
    place_nodes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LAYOUT_OUTPUT_DIR = REPO_ROOT / "layout_output"
CORPUS_DIR = REPO_ROOT / "corpus"

# Box labels the v1 oracle is structurally blind in (non-literal AST text).
# Without a human verdict, "no owner" in these regions means "unknown", not
# "furniture" — so they are never counted furniture on oracle evidence alone.
ORACLE_BLIND_LABELS = frozenset({"table", "display_formula", "formula_number"})

ABSENT = "ABSENT"


@dataclass
class GoldDoc:
    """Node-level gold for one document."""

    doc: str
    # node_id -> set of box keys, or the ABSENT sentinel string
    nodes: dict[str, set[str] | str] = field(default_factory=dict)
    # node_id -> "human" | "audited" | "oracle"
    tier: dict[str, str] = field(default_factory=dict)
    # node_id -> coarse content kind (prose/heading/formula/...)
    kind: dict[str, str] = field(default_factory=dict)
    # box keys confirmed or safely inferred to belong to no node
    furniture: set[str] = field(default_factory=set)
    # box key -> "human" | "oracle"
    furniture_tier: dict[str, str] = field(default_factory=dict)
    # nodes with no usable gold, and why: rejected / unplaced / blind
    ungraded: dict[str, str] = field(default_factory=dict)
    pages: list[dict] = field(default_factory=list)

    def composition(self) -> dict:
        tiers = {"human": 0, "audited": 0, "oracle": 0}
        for t in self.tier.values():
            tiers[t] += 1
        return {
            "n_gold_nodes": len(self.nodes),
            "by_tier": tiers,
            "n_ungraded": len(self.ungraded),
            "n_furniture": len(self.furniture),
            "furniture_human": sum(
                1 for t in self.furniture_tier.values() if t == "human"
            ),
        }


def node_kind(text: str, section_ids: set[str], node_id: str) -> str:
    """Coarse content class, judged from the node text itself.

    The AST's own types are too coarse (doc/section/paragraph), so the classes
    that matter to alignment difficulty — formulas, tables, images — are
    recognised from their Markdown/LaTeX surface.
    """
    t = (text or "").strip()
    if node_id in section_ids:
        return "heading"
    if t.startswith("!["):
        return "image"
    if t.startswith("$$") or t.startswith("$"):
        return "formula"
    if t.startswith("|") or "| --- |" in t:
        return "table"
    if len(_norm(t)) < 4:
        return "marker"
    return "prose"


def _section_ids(ast: dict) -> set[str]:
    out: set[str] = set()

    def walk(n: dict) -> None:
        if n.get("type") == "section" and n.get("id"):
            out.add(n["id"])
        for c in n.get("children") or []:
            walk(c)

    walk(ast)
    return out


def load_pages(doc_dir: Path, granularity: str = "paragraph") -> list[dict]:
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    pages: list[dict] = []
    for entry in manifest["pages"]:
        path = doc_dir / entry["dir"] / f"{granularity}.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        pages.append(
            {
                "page": entry["page"],
                "width": d["width"],
                "height": d["height"],
                "boxes": d.get("boxes", []),
            }
        )
    return pages


def _load_annotations(doc_dir: Path) -> dict:
    path = doc_dir / "annotations.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")).get("items", {})
    return {}


def build_gold(doc: str, granularity: str = "paragraph", dpi: int = 200) -> GoldDoc:
    doc_dir = LAYOUT_OUTPUT_DIR / Path(doc).name
    pdf = CORPUS_DIR / f"{doc_dir.name}.pdf"
    ast = json.loads((doc_dir / "ast.json").read_text(encoding="utf-8"))
    pages = load_pages(doc_dir, granularity)
    ann = _load_annotations(doc_dir)

    gold = GoldDoc(doc=doc_dir.name, pages=pages)
    sections = _section_ids(ast)
    node_texts = dict(_iter_segments(ast))
    for nid, text in node_texts.items():
        gold.kind[nid] = node_kind(text, sections, nid)

    # ---- oracle proposals --------------------------------------------------
    tokens = extract_tokens(pdf, dpi=dpi)
    stream, tokens = build_stream(tokens)
    placements, report = place_nodes(ast, stream)
    owner = gold_token_owner(tokens, placements, report)
    box_tokens = boxes_to_tokens(pages, tokens)

    label_of_box = {
        f"{p['page']}:{i}": (b.get("label") or "unlabeled")
        for p in pages
        for i, b in enumerate(p["boxes"])
    }

    # Majority ownership: box -> node owning >50% of its owned tokens.
    oracle_boxes: dict[str, set[str]] = {}
    boxes_with_owned: set[str] = set()
    for box_key, tks in box_tokens.items():
        counts: dict[str, int] = {}
        for tk in tks:
            nid = owner.get(tk)
            if nid is not None:
                counts[nid] = counts.get(nid, 0) + 1
        if not counts:
            continue
        boxes_with_owned.add(box_key)
        total = sum(counts.values())
        best, n = max(counts.items(), key=lambda kv: kv[1])
        if n * 2 > total:
            oracle_boxes.setdefault(best, set()).add(box_key)

    # ---- assemble, human verdicts winning ----------------------------------
    audit_verdicts = {
        k.split(":", 1)[1]: v for k, v in ann.items() if k.startswith("audit:")
    }
    for nid in node_texts:
        item = ann.get(f"excluded:{nid}")
        if item is not None and item["verdict"] == "boxes":
            gold.nodes[nid] = set(item["boxes"])
            gold.tier[nid] = "human"
            continue
        if item is not None and item["verdict"] == "absent":
            gold.nodes[nid] = ABSENT
            gold.tier[nid] = "human"
            continue

        if nid not in placements:
            gold.ungraded[nid] = "unplaced"
            continue
        verdict = (audit_verdicts.get(nid) or {}).get("verdict")
        if verdict == "reject":
            gold.ungraded[nid] = "rejected"
            continue
        boxes = oracle_boxes.get(nid, set())
        if not boxes:
            # Placed in the stream but its tokens dominate no box (e.g. text
            # swallowed by a bigger region). No box-level claim to grade.
            gold.ungraded[nid] = "no_majority_box"
            continue
        gold.nodes[nid] = boxes
        gold.tier[nid] = "audited" if verdict == "accept" else "oracle"

    # null-queue verdicts: a human either confirmed furniture or assigned an
    # owner the oracle missed. Assignment *extends* gold (the node may also
    # hold oracle boxes when only part of it went missing).
    for key, item in ann.items():
        if not key.startswith("null:"):
            continue
        box_key = key.split(":", 1)[1]
        if item["verdict"] == "furniture":
            gold.furniture.add(box_key)
            gold.furniture_tier[box_key] = "human"
        elif item["verdict"] == "node":
            nid = item["node_id"]
            cur = gold.nodes.get(nid)
            if isinstance(cur, set):
                cur.add(box_key)
            else:
                gold.nodes[nid] = {box_key}
            gold.tier[nid] = "human"
            gold.ungraded.pop(nid, None)

    # Furniture by oracle evidence: boxes where no token has any owner —
    # except in blind labels, where absence of evidence is not evidence.
    claimed = {b for v in gold.nodes.values() if isinstance(v, set) for b in v}
    for box_key in box_tokens:
        if box_key in claimed or box_key in gold.furniture:
            continue
        if box_key in boxes_with_owned:
            continue  # somebody's tokens live here; contested, not furniture
        if label_of_box.get(box_key) in ORACLE_BLIND_LABELS:
            gold.ungraded.setdefault(f"box:{box_key}", "blind")
            continue
        gold.furniture.add(box_key)
        gold.furniture_tier[box_key] = "oracle"

    return gold
