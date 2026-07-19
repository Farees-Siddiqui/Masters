"""Real-document benchmark: score the aligners against text-layer gold.

Unlike :mod:`benchmark.synthetic`, nothing here is constructed. The PDF boxes are
whatever PP-DocLayoutV3/PP-OCRv5 actually detected, the AST is whatever Mistral
actually returned, and gold comes from the PDF's own text layer via
:mod:`benchmark.oracle` — a referee neither engine can influence. This is the
cross-engine disagreement the deployed system faces; the synthetic sweep only
ever measured robustness to IID character noise.

Everything runs off the on-disk layout cache (``layout_output/<doc>/``), so no
Paddle inference and no Mistral API call is needed to reproduce a run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from alignment.naive_aligner import _iter_segments

from .oracle import _norm, derive_gold
from .token_metrics import (
    StratifiedScores,
    macro_f1,
    merge_scores,
    predict_token_owner,
    score_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_pages(doc_dir: Path, granularity: str = "paragraph") -> list[dict]:
    """Load cached layout boxes for every page, in page order."""
    manifest = json.loads((doc_dir / "manifest.json").read_text(encoding="utf-8"))
    pages: list[dict] = []
    for entry in manifest["pages"]:
        p = entry["page"]
        path = doc_dir / entry["dir"] / f"{granularity}.json"
        if not path.is_file():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        pages.append(
            {
                "page": p,
                "width": d["width"],
                "height": d["height"],
                "boxes": d.get("boxes", []),
            }
        )
    return pages


def token_labels(pages: list[dict], box_tokens: dict[str, list[str]]) -> dict[str, str]:
    """Map each token to the layout label of its box — the stratum it's scored in."""
    label_of_box = {
        f"{page['page']}:{i}": b.get("label") or "unlabeled"
        for page in pages
        for i, b in enumerate(page.get("boxes", []))
    }
    out: dict[str, str] = {}
    for box_key, tks in box_tokens.items():
        lab = label_of_box.get(box_key, "unlabeled")
        for tk in tks:
            out[tk] = lab
    return out


def score_doc(
    pdf_path: Path,
    doc_dir: Path,
    methods: dict[str, object],
    granularity: str = "paragraph",
    dpi: int = 200,
) -> tuple[dict, dict[str, StratifiedScores]]:
    """Derive gold for one doc, score every method, return (summary, raw scores).

    The raw :class:`StratifiedScores` come back alongside the JSON-able summary
    so the caller can pool counts across documents without re-deriving anything.
    """
    ast = json.loads((doc_dir / "ast.json").read_text(encoding="utf-8"))
    pages = load_pages(doc_dir, granularity)

    tokens, token_gold, box_tokens, report = derive_gold(pdf_path, ast, pages, dpi=dpi)
    labels = token_labels(pages, box_tokens)
    node_text = {nid: _norm(t) for nid, t in _iter_segments(ast)}

    n_boxes = sum(len(p["boxes"]) for p in pages)
    results = []
    raw: dict[str, StratifiedScores] = {}
    for name, fn in methods.items():
        t0 = time.perf_counter()
        out = fn(ast, pages)
        secs = time.perf_counter() - t0
        pred = predict_token_owner(out.get("reverse", {}), box_tokens)
        # Pass the oracle's placed-node set so predictions of nodes the oracle
        # itself failed to place are excluded as unverifiable, not charged as
        # false attributions (see score_tokens for the accepted asymmetry).
        scores = score_tokens(
            token_gold, pred, labels, node_text, placed_nodes=report.placed_node_ids
        )
        raw[name] = scores
        results.append({"method": name, "secs": round(secs, 2), **scores.to_dict()})

    summary = {
        "doc": doc_dir.name,
        "pdf": pdf_path.name,
        "n_pages": len(pages),
        "n_boxes": n_boxes,
        "n_tokens": len(tokens),
        "oracle": report.to_dict(),
        "results": results,
    }
    return summary, raw


def run_real(
    pdf_path: Path,
    doc_dir: Path,
    methods: dict[str, object],
    granularity: str = "paragraph",
    dpi: int = 200,
) -> dict:
    """Single-document run (kept for `--doc`)."""
    summary, _ = score_doc(pdf_path, doc_dir, methods, granularity, dpi)
    return {
        "granularity": granularity,
        "dpi": dpi,
        "n_docs": 1,
        "docs": [summary],
        **summary,
    }


def run_corpus(
    docs: list[tuple[str, Path, Path]],
    methods: dict[str, object],
    granularity: str = "paragraph",
    dpi: int = 200,
    log=print,
) -> dict:
    """Score every doc, then pool. ``docs`` is ``[(name, pdf_path, doc_dir)]``.

    Reports **micro** (token-pooled) and **macro** (per-document mean) F1 side by
    side: a single corpus number averages away exactly the per-document variance
    that says whether one paper is carrying the result.
    """
    per_doc: list[dict] = []
    raw_by_method: dict[str, list] = {m: [] for m in methods}
    raw_by_method_label: dict[str, dict[str, list]] = {m: {} for m in methods}
    t_start = time.perf_counter()

    for i, (name, pdf, doc_dir) in enumerate(docs, 1):
        t0 = time.perf_counter()
        try:
            summary, raw = score_doc(pdf, doc_dir, methods, granularity, dpi)
        except Exception as exc:
            log(f"  [{i}/{len(docs)}] {name:<14} FAILED: {type(exc).__name__}: {exc}")
            continue
        per_doc.append(summary)
        for m, s in raw.items():
            raw_by_method[m].append(s.overall)
            for lab, sc in s.by_label.items():
                raw_by_method_label[m].setdefault(lab, []).append(sc)
        o = summary["oracle"]
        best = ", ".join(
            f"{r['method']} F1 {r['overall']['f1']:.3f}" if r["overall"]["f1"] is not None else f"{r['method']} F1 —"
            for r in summary["results"]
        )
        log(
            f"  [{i}/{len(docs)}] {name:<14} {summary['n_pages']:>2}p "
            f"{summary['n_tokens']:>5} tok  placed {o['placement_rate']:.0%}  "
            f"{best}  ({time.perf_counter()-t0:.0f}s)"
        )

    # Pool: micro over tokens, macro over documents.
    aggregate = []
    for m in methods:
        micro = merge_scores(raw_by_method[m])
        by_label = {
            lab: merge_scores(parts).to_dict()
            for lab, parts in raw_by_method_label[m].items()
        }
        aggregate.append({
            "method": m,
            "secs": round(sum(
                r["secs"] for d in per_doc for r in d["results"] if r["method"] == m
            ), 2),
            "overall": micro.to_dict(),
            "macro_f1": (lambda v: round(v, 4) if v is not None else None)(macro_f1(raw_by_method[m])),
            "by_label": dict(sorted(by_label.items())),
        })

    return {
        "corpus": "arxiv",
        "granularity": granularity,
        "dpi": dpi,
        "n_docs": len(per_doc),
        "n_pages": sum(d["n_pages"] for d in per_doc),
        "n_boxes": sum(d["n_boxes"] for d in per_doc),
        "n_tokens": sum(d["n_tokens"] for d in per_doc),
        "wall_secs": round(time.perf_counter() - t_start, 1),
        "oracle": {
            "n_nodes": sum(d["oracle"]["n_nodes"] for d in per_doc),
            "n_placed": sum(d["oracle"]["n_placed"] for d in per_doc),
            "n_unplaced": sum(d["oracle"]["n_unplaced"] for d in per_doc),
            "n_tokens": sum(d["oracle"]["n_tokens"] for d in per_doc),
            "n_gold_tokens": sum(d["oracle"]["n_gold_tokens"] for d in per_doc),
            "n_null_tokens": sum(d["oracle"]["n_null_tokens"] for d in per_doc),
            "n_conflicts": sum(d["oracle"]["n_conflicts"] for d in per_doc),
            "placement_rate": round(
                sum(d["oracle"]["n_placed"] for d in per_doc)
                / max(sum(d["oracle"]["n_nodes"] for d in per_doc), 1), 4),
            "token_coverage": round(
                sum(d["oracle"]["n_gold_tokens"] for d in per_doc)
                / max(sum(d["oracle"]["n_tokens"] for d in per_doc), 1), 4),
            "unplaced_samples": [
                s for d in per_doc for s in d["oracle"]["unplaced_samples"][:2]
            ][:10],
        },
        "results": aggregate,
        "docs": per_doc,
    }


def print_real(data: dict) -> None:
    o = data["oracle"]
    n_docs = data.get("n_docs", 1)
    scope = (
        f"corpus '{data.get('corpus')}': {n_docs} documents"
        if n_docs > 1
        else f"doc={data.get('doc')}"
    )
    print(
        f"\n{scope}  ·  {data['n_pages']} pages  ·  {data['n_boxes']} boxes  ·  "
        f"{data['n_tokens']:,} text-layer tokens"
    )
    print(
        f"oracle: placed {o['n_placed']}/{o['n_nodes']} nodes ({o['placement_rate']:.1%}), "
        f"token coverage {o['token_coverage']:.1%}, "
        f"{o['n_null_tokens']} null tokens, {o['n_conflicts']} conflicts"
    )
    if n_docs > 1 and data.get("docs"):
        rates = sorted((d["oracle"]["placement_rate"], d["doc"]) for d in data["docs"])
        print(
            f"        placement spread: {rates[0][0]:.0%} ({rates[0][1]}) .. "
            f"{rates[-1][0]:.0%} ({rates[-1][1]}) — low = maths-heavy, gold is thinner there"
        )

    # Per-document F1 — a single corpus number hides which doc carries the result.
    if n_docs > 1 and data.get("docs"):
        methods = [r["method"] for r in data["results"]]
        print(f"\nper-document F1 ({n_docs} docs)")
        head = f"{'doc':<14}{'pages':>6}{'tokens':>8}{'placed':>8}" + "".join(f"{m:>13}" for m in methods)
        print(head)
        print("-" * len(head))
        for d in data["docs"]:
            row = (
                f"{d['doc']:<14}{d['n_pages']:>6}{d['n_tokens']:>8}"
                f"{d['oracle']['placement_rate']:>7.0%} "
            )
            for m in methods:
                r = next((x for x in d["results"] if x["method"] == m), None)
                v = r["overall"]["f1"] if r else None
                row += f"{v:>13.3f}" if v is not None else f"{'—':>13}"
            print(row)

    def fmt(v: float | None, w: int = 8) -> str:
        return f"{v:>{w}.3f}" if v is not None else f"{'—':>{w}}"

    # micro = tokens pooled (long docs pull harder); macro = per-doc mean.
    hdr = (
        f"{'method':<12}{'P':>8}{'R':>8}{'F1(µ)':>8}{'F1(M)':>8}{'acc':>8}"
        f"{'null.acc':>10}{'false.attr':>12}{'excl':>7}{'secs':>8}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in data["results"]:
        ov = r["overall"]
        print(
            f"{r['method']:<12}{fmt(ov['precision'])}{fmt(ov['recall'])}{fmt(ov['f1'])}"
            f"{fmt(r.get('macro_f1'))}{fmt(ov['accuracy'])}"
            f"{fmt(ov['null_accuracy'], 10)}{fmt(ov['false_attribution_rate'], 12)}"
            f"{ov['n_excluded']:>7}{r['secs']:>8.2f}"
        )
    if n_docs > 1:
        print("  F1(µ) = micro, tokens pooled across docs · F1(M) = macro, per-document mean")

    # Per-stratum, because the aggregate hides which regions actually fail.
    # Strata split into two kinds and must be read with different metrics:
    # regions that carry prose (score F1) and regions that are pure page
    # furniture (no gold owner exists — score abstention).
    labels = sorted({l for r in data["results"] for l in r["by_label"]})
    methods = [r["method"] for r in data["results"]]

    def stratum_stats(lab: str) -> dict:
        return data["results"][0]["by_label"].get(lab, {})

    # Split by MAJORITY, not by "has any gold owner". Pooled over enough docs a
    # figure region picks up a stray owned token, which would file it under prose
    # and print a meaningless F1 of 0.000 for what is really page furniture.
    def is_furniture(lab: str) -> bool:
        s = stratum_stats(lab)
        owned, null = s.get("n_gold_owned", 0), s.get("n_gold_null", 0)
        return null > owned

    prose = [l for l in labels if not is_furniture(l) and stratum_stats(l).get("n_gold_owned", 0) > 0]
    furniture = [l for l in labels if is_furniture(l)]

    print("\nprose strata — F1 (n = scored tokens)")
    head = f"{'label':<20}{'n':>7}" + "".join(f"{m:>13}" for m in methods)
    print(head)
    print("-" * len(head))
    for lab in prose:
        s0 = stratum_stats(lab)
        n = s0.get("n_tokens", 0) - s0.get("n_excluded", 0)
        row = f"{lab:<20}{n:>7}"
        for r in data["results"]:
            s = r["by_label"].get(lab)
            row += fmt(s["f1"], 13) if s else f"{'—':>13}"
        print(row)

    print("\nfurniture strata — abstention accuracy (no node owns these; 1.0 = correctly ignored)")
    print(head)
    print("-" * len(head))
    for lab in furniture:
        s0 = stratum_stats(lab)
        n = s0.get("n_tokens", 0) - s0.get("n_excluded", 0)
        row = f"{lab:<20}{n:>7}"
        for r in data["results"]:
            s = r["by_label"].get(lab)
            row += fmt(s["null_accuracy"], 13) if s else f"{'—':>13}"
        print(row)
