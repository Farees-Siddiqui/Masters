"""Alignment benchmark CLI.

Two modes, answering different questions:

    # Controlled stress: exact gold by construction, one perturbation at a time.
    python -m benchmark synthetic --methods stream,similarity \
        --noise 0,0.05,0.1,0.15,0.2,0.3 --split 60

    # Real documents: cross-engine gold from the PDF text layer, stratified.
    python -m benchmark real --doc resnet

``synthetic`` builds its boxes *from* the AST, so it measures robustness to
character noise but cannot represent the real task (two independent OCR engines)
or the NULL class. ``real`` scores against a text-layer oracle on real
PP-OCRv5 boxes — see :mod:`benchmark.real`. Both write JSON for the dashboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from alignment import align_cpsat, align_similarity, align_stream

from .corpus import CORPORA, build_corpus, built_docs, fetch_pdf, pdf_path
from .metrics import score_alignment
from .real import print_real, run_corpus, run_real
from .synthetic import build_pages

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AST = REPO_ROOT / "layout_output" / "resnet" / "ast.json"
LAYOUT_OUTPUT = REPO_ROOT / "layout_output"
CORPUS_DIR = REPO_ROOT / "corpus"

# A 2x2 over (representation) x (aggregation), so the two hypotheses for
# `similarity`'s weakness stay separable: is the embedding the wrong signal, or
# is unconstrained argmax the wrong way to combine it?
#
#                    argmax                 CP-SAT constraints
#   embedding        similarity             cpsat
#   char n-gram      lexical                cpsat-lex
#
# `lexical` runs through the same solver with the constraints switched off
# (no ordering, no length cap), which reduces exactly to per-box argmax — so any
# gap to `cpsat-lex` is attributable to the constraints alone, not to a second
# code path.
_UNCONSTRAINED = dict(monotone=False, max_len_ratio=1e9)

METHODS = {
    "stream": lambda ast, pages: align_stream(ast, pages, "paragraph"),
    "similarity": lambda ast, pages: align_similarity(ast, pages, "paragraph"),
    "lexical": lambda ast, pages: align_cpsat(ast, pages, "paragraph", scorer="lexical", **_UNCONSTRAINED),
    "cpsat": lambda ast, pages: align_cpsat(ast, pages, "paragraph", scorer="embed"),
    "cpsat-lex": lambda ast, pages: align_cpsat(ast, pages, "paragraph", scorer="lexical"),
}


def _run_synthetic(args: argparse.Namespace) -> list[dict]:
    ast = json.loads(Path(args.ast).read_text(encoding="utf-8"))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(METHODS)}")
    noises = [float(x) for x in args.noise.split(",") if x.strip()]

    rows: list[dict] = []
    for noise in noises:
        # Same corrupted pages for every method at this noise level (fair compare).
        rng = random.Random(args.seed + int(round(noise * 1000)))
        pages, gold_owner = build_pages(
            ast, noise, rng, split_width=args.split, shuffle=args.shuffle
        )
        n_boxes = sum(len(p["boxes"]) for p in pages)
        for method in methods:
            t0 = time.perf_counter()
            result = METHODS[method](ast, pages)
            secs = time.perf_counter() - t0
            s = score_alignment(gold_owner, result.get("reverse", {}))
            rows.append(
                {
                    "method": method,
                    "noise": noise,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                    "owner_acc": s.owner_accuracy,
                    "n_boxes": n_boxes,
                    "secs": secs,
                }
            )
    return rows


def _methods_from(args: argparse.Namespace) -> dict:
    names = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in names:
        if m not in METHODS:
            raise SystemExit(f"unknown method {m!r}; choose from {sorted(METHODS)}")
    return {m: METHODS[m] for m in names}


def _run_real(args: argparse.Namespace) -> dict:
    methods = _methods_from(args)

    if args.all:
        docs = CORPORA[args.corpus]
        ready = built_docs(docs)
        missing = [d.name for d in docs if d not in ready]
        if not ready:
            raise SystemExit(
                f"no built docs in corpus '{args.corpus}' — run `python -m benchmark fetch` first"
            )
        if missing:
            print(f"note: skipping {len(missing)} unbuilt doc(s): {', '.join(missing)}")
        print(f"scoring {len(ready)}/{len(docs)} docs from corpus '{args.corpus}'\n")
        triples = [(d.name, pdf_path(d), LAYOUT_OUTPUT / d.name) for d in ready]
        data = run_corpus(triples, methods, args.granularity, args.dpi)
        data["corpus"] = args.corpus
        return data

    doc_dir = LAYOUT_OUTPUT / Path(args.doc).name
    if not doc_dir.is_dir():
        raise SystemExit(f"no layout cache at {doc_dir} — run the layout tab on the PDF first")
    # Prefer the corpus copy; fall back to a PDF sitting at the repo root.
    candidates = [Path(args.pdf)] if args.pdf else [
        CORPUS_DIR / f"{doc_dir.name}.pdf",
        REPO_ROOT / f"{doc_dir.name}.pdf",
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        raise SystemExit(f"source PDF not found (tried: {', '.join(map(str, candidates))})")

    return run_real(
        pdf_path=src,
        doc_dir=doc_dir,
        methods=methods,
        granularity=args.granularity,
        dpi=args.dpi,
    )


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'method':<12}{'noise':>7}{'P':>8}{'R':>8}{'F1':>8}{'own.acc':>9}{'boxes':>8}{'secs':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['method']:<12}{r['noise']:>7.2f}{r['precision']:>8.3f}{r['recall']:>8.3f}"
            f"{r['f1']:>8.3f}{r['owner_acc']:>9.3f}{r['n_boxes']:>8}{r['secs']:>8.2f}"
        )


def _write_outputs(rows: list[dict], json_path: str | None, csv_path: str | None) -> None:
    if json_path:
        Path(json_path).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {json_path}")
    if csv_path and rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmark", description="Alignment benchmark")
    sub = parser.add_subparsers(dest="mode")

    syn = sub.add_parser("synthetic", help="synthetic OCR-noise sweep (no download)")
    syn.add_argument("--ast", default=str(DEFAULT_AST), help="AST json to source text from")
    syn.add_argument("--methods", default="stream,similarity", help="comma-separated aligners")
    syn.add_argument("--noise", default="0,0.05,0.1,0.15,0.2,0.3", help="comma-separated noise rates")
    syn.add_argument("--split", type=int, default=60, help="char width to split paragraphs into box 'lines' (0 = whole segment)")
    syn.add_argument("--shuffle", action="store_true", help="permute box order (model multi-column / reading-order disruption)")
    syn.add_argument("--seed", type=int, default=0)
    syn.add_argument("--json", dest="json_path", default=None)
    syn.add_argument("--csv", dest="csv_path", default=None)

    fetch = sub.add_parser("fetch", help="download + build the pinned benchmark corpus")
    fetch.add_argument("--corpus", default="arxiv", choices=sorted(CORPORA))
    fetch.add_argument("--only-fetch", action="store_true", help="download PDFs, skip OCR/layout")
    fetch.add_argument("--limit", type=int, default=0, help="build at most N docs (0 = all)")

    real = sub.add_parser("real", help="real-document benchmark vs. PDF text-layer gold")
    real.add_argument("--doc", default="resnet", help="layout_output/<doc> cache to score")
    real.add_argument("--all", action="store_true", help="score every built doc in --corpus")
    real.add_argument("--corpus", default="arxiv", choices=sorted(CORPORA))
    real.add_argument("--pdf", default=None, help="source PDF (default: <repo>/<doc>.pdf)")
    real.add_argument("--methods", default="stream,similarity", help="comma-separated aligners")
    real.add_argument("--granularity", default="paragraph", choices=["paragraph", "line", "word"])
    real.add_argument("--dpi", type=int, default=200, help="DPI the layout cache was rendered at")
    real.add_argument(
        "--json",
        dest="json_path",
        default=str(REPO_ROOT / "results_real.json"),
        help="where to write results (default: repo root)",
    )

    args = parser.parse_args()
    if args.mode == "fetch":
        docs = CORPORA[args.corpus]
        if args.limit:
            docs = docs[: args.limit]
        print(f"corpus '{args.corpus}': {len(docs)} docs -> {CORPUS_DIR}")
        if args.only_fetch:
            for d in docs:
                fetch_pdf(d)
        else:
            rows = build_corpus(docs)
            ok = [r for r in rows if r["ok"]]
            print(f"\n{len(ok)}/{len(rows)} docs ready, {sum(r.get('pages', 0) for r in ok)} pages")
    elif args.mode == "real":
        data = _run_real(args)
        print_real(data)
        if args.json_path:
            Path(args.json_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"\nwrote {args.json_path}")
    elif args.mode in (None, "synthetic"):
        # default to synthetic when no subcommand given
        if args.mode is None:
            args = syn.parse_args([])
        rows = _run_synthetic(args)
        _print_table(rows)
        _write_outputs(rows, args.json_path, args.csv_path)
    else:
        raise SystemExit(f"unknown mode {args.mode!r}")


if __name__ == "__main__":
    main()
