"""CLI for the node-level localization benchmark.

    python -m benchmark_v2_0 --docs resnet --methods stream,similarity
    python -m benchmark_v2_0 --all --out results_v2.json

Runs off the on-disk layout cache; no Paddle inference or API calls.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark.__main__ import METHODS

from .gold import CORPUS_DIR, LAYOUT_OUTPUT_DIR, build_gold
from .score import ScoreCard, merge_cards, score_doc


def _available_docs() -> list[str]:
    return sorted(
        p.stem
        for p in CORPUS_DIR.glob("*.pdf")
        if (LAYOUT_OUTPUT_DIR / p.stem / "ast.json").is_file()
    )


def main() -> None:
    ap = argparse.ArgumentParser(prog="benchmark_v2_0", description=__doc__)
    ap.add_argument("--docs", default="", help="comma-separated doc names")
    ap.add_argument("--all", action="store_true", help="every doc with a cached layout")
    ap.add_argument("--methods", default="stream,similarity")
    ap.add_argument("--granularity", default="paragraph")
    ap.add_argument("--human-only", action="store_true",
                    help="score only human-tier gold (skip unverified oracle proposals)")
    ap.add_argument("--out", default="", help="write full JSON results here")
    args = ap.parse_args()

    docs = _available_docs() if args.all else [d for d in args.docs.split(",") if d]
    if not docs:
        ap.error("pick --docs or --all; available: " + ", ".join(_available_docs()))
    methods = {m: METHODS[m] for m in args.methods.split(",") if m}

    per_doc: list[dict] = []
    cards_by_method: dict[str, list[ScoreCard]] = {m: [] for m in methods}
    compositions: list[dict] = []

    for i, doc in enumerate(docs, 1):
        t0 = time.perf_counter()
        gold = build_gold(doc, args.granularity)
        if args.human_only:
            keep = {n for n, t in gold.tier.items() if t == "human"}
            gold.nodes = {n: v for n, v in gold.nodes.items() if n in keep}
            gold.tier = {n: t for n, t in gold.tier.items() if n in keep}
            gold.furniture = {
                b for b in gold.furniture if gold.furniture_tier.get(b) == "human"
            }
        comp = gold.composition()
        compositions.append(comp)

        ast = json.loads(
            (LAYOUT_OUTPUT_DIR / doc / "ast.json").read_text(encoding="utf-8")
        )
        doc_result: dict = {"doc": doc, "gold": comp, "results": {}}
        for name, fn in methods.items():
            out = fn(ast, gold.pages)
            card = score_doc(gold, out.get("reverse", {}))
            cards_by_method[name].append(card)
            doc_result["results"][name] = card.to_dict()
        per_doc.append(doc_result)
        acc = ", ".join(
            f"{m} {v['localization_accuracy']}" for m, v in doc_result["results"].items()
        )
        print(
            f"[{i}/{len(docs)}] {doc:<14} gold {comp['n_gold_nodes']:>3} nodes "
            f"(h{comp['by_tier']['human']}/a{comp['by_tier']['audited']}"
            f"/o{comp['by_tier']['oracle']})  {acc}  "
            f"({time.perf_counter() - t0:.0f}s)"
        )

    # ---- corpus table ------------------------------------------------------
    print(f"\nnode-level localization — {len(docs)} doc(s), IoU >= 0.5")
    total_h = sum(c["by_tier"]["human"] for c in compositions)
    total_a = sum(c["by_tier"]["audited"] for c in compositions)
    total_o = sum(c["by_tier"]["oracle"] for c in compositions)
    total = total_h + total_a + total_o
    if total:
        print(
            f"gold provenance: {total_h} human, {total_a} audited, {total_o} "
            f"unverified-oracle of {total} nodes "
            f"({(total_h + total_a) / total:.0%} verified)"
        )

    hdr = f"{'method':<12}{'loc.acc':>9}{'meanIoU':>9}{'exact':>7}{'missed':>8}{'absent.acc':>12}{'false.claim':>13}"
    print("\n" + hdr)
    print("-" * len(hdr))
    agg: dict[str, ScoreCard] = {}
    for m, cards in cards_by_method.items():
        card = merge_cards(cards)
        agg[m] = card

        def fmt(v: float | None, w: int) -> str:
            return f"{v:>{w}.3f}" if v is not None else f"{'—':>{w}}"

        print(
            f"{m:<12}{fmt(card.localization_accuracy, 9)}{fmt(card.mean_iou, 9)}"
            f"{card.n_exact:>7}{card.n_missed:>8}{fmt(card.absent_accuracy, 12)}"
            f"{fmt(card.false_claim_rate, 13)}"
        )

    # Per-kind: the sentence that says what to work on next.
    kinds = sorted({k for c in agg.values() for k in c.by_kind})
    if kinds:
        print(f"\nby node kind — localization accuracy (n)")
        head = f"{'kind':<12}" + "".join(f"{m:>18}" for m in agg)
        print(head)
        print("-" * len(head))
        for k in kinds:
            row = f"{k:<12}"
            for card in agg.values():
                s = card.by_kind.get(k)
                row += (
                    f"{s['localization_accuracy']:>11.3f} ({s['n']:>4})" if s else f"{'—':>18}"
                )
            print(row)

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "benchmark": "v2.0-node-localization",
                    "iou_threshold": 0.5,
                    "granularity": args.granularity,
                    "human_only": args.human_only,
                    "docs": per_doc,
                    "aggregate": {m: c.to_dict() for m, c in agg.items()},
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
