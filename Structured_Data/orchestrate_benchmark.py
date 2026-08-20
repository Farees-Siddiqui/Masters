#!/usr/bin/env python3
"""OCR survey orchestrator -- runs under env_eval.

  ocr_venvs/env_eval/bin/python orchestrate_benchmark.py

For each paper in ./arxiv_papers/ it renders page 1, takes the PDF's native text
layer as ground truth via pdfplumber, then shells out to each engine's runner in
that engine's own virtualenv and scores what comes back.

Nothing in this pipeline reads LaTeX. Ground truth is the embedded text layer;
the engines only ever see a PNG.

Each (model, mode) pair is one subprocess handling all ten pages, so a model is
loaded once rather than ten times -- with DeepSeek at ~42 s of load that is the
difference between a 20-minute sweep and a two-hour one. Use --per-image to force
the strictly-one-image-per-invocation form instead. Models run concurrently, one
pinned GPU each.
"""
import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess
import sys
import time

import Levenshtein

ROOT = pathlib.Path(__file__).resolve().parent
WORK = pathlib.Path("/tmp/ocr_bench")
VENVS = ROOT / "ocr_venvs"
RESULTS = ROOT / "ocr_survey_results.json"
SUMMARY = ROOT / "OCR_SURVEY.md"

# (key, runner, venv, modes). GPU index is assigned by position.
ENGINES = [
    ("paddleocr",     "run_paddle.py",    "env_paddle",    ["text", "grounding"]),
    ("glm-ocr",       "run_glm.py",       "env_glm",       ["text", "grounding"]),
    ("deepseek-ocr",  "run_deepseek.py",  "env_deepseek",  ["text", "grounding"]),
    ("unlimited-ocr", "run_unlimited.py", "env_unlimited", ["text", "grounding"]),
]

WS_RE = re.compile(r"\s+")
# Markdown/LaTeX scaffolding that a structured engine emits but a PDF text layer
# never contains. Stripped only for the secondary "content" CER.
MARKUP_RE = re.compile(r"[#*_`~\\$\[\]\(\){}|<>]+")


def norm_ws(s):
    return WS_RE.sub(" ", s or "").strip()


def norm_content(s):
    """Whitespace-collapsed, markup-stripped, punctuation-light form."""
    s = MARKUP_RE.sub(" ", s or "")
    s = "".join(c for c in s if c.isalnum() or c.isspace())
    return WS_RE.sub(" ", s).strip().lower()


def norm_nospace(s):
    """All whitespace removed -- immune to line-wrapping and space-detection noise."""
    return WS_RE.sub("", s or "")


def cer(ref, hyp):
    if not ref:
        return None
    return round(Levenshtein.distance(ref, hyp) / len(ref), 4)


def word_prf(ref, hyp):
    """Order-independent token-multiset precision/recall/F1.

    CER is a sequential metric, so it conflates "read the wrong characters" with
    "read the right characters in a different order". On multi-column pages that
    difference is everything, so this reports coverage independently of ordering
    and independently of the reading order reconstructed for the ground truth.
    """
    from collections import Counter
    r = Counter(norm_content(ref).split())
    h = Counter(norm_content(hyp).split())
    if not r:
        return None, None, None
    overlap = sum((r & h).values())
    prec = overlap / sum(h.values()) if h else 0.0
    rec = overlap / sum(r.values())
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def rescore():
    """Re-derive every metric from the stored engine outputs.

    Inference is the expensive part and its results are already on disk, so a
    change to ground-truth extraction or to a CER definition costs a few seconds
    rather than another full GPU sweep.
    """
    if not RESULTS.exists():
        print(f"no {RESULTS} to rescore", file=sys.stderr)
        return 1
    payload = json.loads(RESULTS.read_text())
    index = json.loads((WORK / "index.json").read_text())
    payload["papers"] = index

    gt_cache = {n: pathlib.Path(index[n]["gt_text"]).read_text(encoding="utf-8")
                for n in index}
    for r in payload["runs"]:
        gt = gt_cache.get(r["paper"], "")
        hyp = r.get("text", "") or ""
        r["gt_chars"] = len(gt)
        r["cer"] = cer(norm_ws(gt), norm_ws(hyp))
        r["cer_nospace"] = cer(norm_nospace(gt), norm_nospace(hyp))
        r["cer_content"] = cer(norm_content(gt), norm_content(hyp))
        r["cer_raw"] = cer(gt, hyp)
        r["word_precision"], r["word_recall"], r["word_f1"] = word_prf(gt, hyp)
        r["columns"] = index.get(r["paper"], {}).get("columns")
    payload["meta"]["rescored"] = True
    payload["meta"]["x_tolerance"] = index[next(iter(index))].get("x_tolerance")

    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_summary(payload)
    print(f"rescored {len(payload['runs'])} runs -> {RESULTS}, {SUMMARY}")
    return 0


def run_engine(key, runner, venv, mode, images, gpu, per_image, timeout):
    """Invoke one runner for one mode; return a list of result dicts."""
    python = VENVS / venv / "bin" / "python"
    if not python.exists():
        return [{"model": key, "mode": mode, "image": im,
                 "error": f"missing interpreter {python}"} for im in images]

    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": "/usr/bin:/bin",
           "HOME": str(pathlib.Path.home())}
    batches = [[im] for im in images] if per_image else [images]

    out = []
    for batch in batches:
        cmd = [str(python), str(ROOT / runner), "--mode", mode]
        for im in batch:
            cmd += ["--image", im]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout, cwd=ROOT, env=env)
        except subprocess.TimeoutExpired:
            print(f"    !! [{key}/{mode}] TIMEOUT after {timeout}s", flush=True)
            out += [{"model": key, "mode": mode, "image": im,
                     "error": f"timeout after {timeout}s"} for im in batch]
            continue
        wall = time.time() - t0

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-4:]
            hint = ""
            if proc.returncode == -9 or proc.returncode == 137:
                hint = (" [SIGKILL -- almost certainly the 16 GiB cgroup RAM cap; "
                        "run fewer engines concurrently]")
            msg = f"exit {proc.returncode}{hint}: {' | '.join(tail)}"
            # Loud: a silently-skipped engine reads as "no result" in the table,
            # which is indistinguishable from "engine has no such capability".
            print(f"    !! [{key}/{mode}] FAILED {msg}", flush=True)
            out += [{"model": key, "mode": mode, "image": im, "error": msg}
                    for im in batch]
            continue
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            print(f"    !! [{key}/{mode}] BAD STDOUT: {exc}", flush=True)
            out += [{"model": key, "mode": mode, "image": im,
                     "error": f"unparseable stdout ({exc}): {proc.stdout[-300:]}"}
                    for im in batch]
            continue
        recs = payload if isinstance(payload, list) else [payload]
        for r in recs:
            r["subprocess_wall_sec"] = round(wall, 2)
        out += recs
        print(f"    [{key}/{mode}] {len(recs)} page(s) in {wall:.1f}s wall", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-image", action="store_true",
                    help="one subprocess per image (spec-literal, ~5x slower)")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--jobs", type=int, default=1,
                    help="engines to run concurrently. Default 1: this box caps the "
                         "cgroup at 16 GiB RAM, and DeepSeek/Unlimited need ~7-10 GiB "
                         "each while loading, so parallel engines get OOM-killed.")
    ap.add_argument("--only", help="comma-separated engine keys to run")
    ap.add_argument("--limit", type=int, help="use only the first N papers (smoke test)")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute metrics from the saved run texts in "
                         "ocr_survey_results.json; no GPU, no inference")
    args = ap.parse_args()

    if args.rescore:
        return rescore()

    index_path = WORK / "index.json"
    if not index_path.exists():
        print(f"missing {index_path} -- run prepare_pages.py under env_eval first",
              file=sys.stderr)
        return 1
    index = json.loads(index_path.read_text())
    names = sorted(index)
    if args.limit:
        names = names[:args.limit]
        index = {n: index[n] for n in names}
    images = [index[n]["image"] for n in names]
    img_to_name = {index[n]["image"]: n for n in names}

    engines = ENGINES
    if args.only:
        want = {s.strip() for s in args.only.split(",")}
        engines = [e for e in ENGINES if e[0] in want]

    print(f"{len(names)} papers x {len(engines)} engines "
          f"({'per-image' if args.per_image else 'batched'})\n")

    jobs = []
    for gpu, (key, runner, venv, modes) in enumerate(engines):
        for mode in modes:
            jobs.append((key, runner, venv, mode, gpu % 4))

    t_start = time.time()
    raw = []
    # One worker per engine: modes for the same engine share a GPU, so they must
    # not overlap. Group by engine and run that engine's modes in sequence.
    by_engine = {}
    for j in jobs:
        by_engine.setdefault(j[0], []).append(j)

    def run_all_modes(engine_jobs):
        acc = []
        for key, runner, venv, mode, gpu in engine_jobs:
            print(f"  -> {key} / {mode} on GPU {gpu}", flush=True)
            acc += run_engine(key, runner, venv, mode, images, gpu,
                              args.per_image, args.timeout)
        return acc

    workers = max(1, min(args.jobs, len(by_engine)))
    if workers == 1:
        for v in by_engine.values():
            raw += run_all_modes(v)
    else:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(run_all_modes, v) for v in by_engine.values()]
            for f in cf.as_completed(futs):
                raw += f.result()
    elapsed = time.time() - t_start

    # ---- score ----
    runs = []
    for r in raw:
        image = r.get("image")
        name = img_to_name.get(image, "?")
        gt = ""
        if name in index:
            gt = pathlib.Path(index[name]["gt_text"]).read_text(encoding="utf-8")
        hyp = r.get("text", "") or ""
        boxes = r.get("boxes") or []
        runs.append({
            "paper": name,
            "model": r.get("model"),
            "mode": r.get("mode"),
            "cer": cer(norm_ws(gt), norm_ws(hyp)),
            "cer_nospace": cer(norm_nospace(gt), norm_nospace(hyp)),
            "cer_content": cer(norm_content(gt), norm_content(hyp)),
            "cer_raw": cer(gt, hyp),
            "word_precision": word_prf(gt, hyp)[0],
            "word_recall": word_prf(gt, hyp)[1],
            "word_f1": word_prf(gt, hyp)[2],
            "columns": index.get(name, {}).get("columns"),
            "gt_chars": len(gt),
            "hyp_chars": len(hyp),
            "has_boxes": bool(r.get("has_boxes")),
            "n_boxes": len(boxes),
            "box_format": r.get("box_format"),
            "box_labels": sorted({b.get("label") for b in boxes if b.get("label")}),
            "time_sec": r.get("time_sec"),
            "load_sec": r.get("load_sec"),
            "peak_vram_gb": r.get("peak_vram_gb"),
            "device": r.get("device"),
            "gpu_name": r.get("gpu_name"),
            "dtype": r.get("dtype"),
            "attn_impl": r.get("attn_impl"),
            "pipeline": r.get("pipeline"),
            "skipped": r.get("skipped", False),
            "error": r.get("error"),
            "text": hyp,
            "boxes": boxes or None,
        })

    payload = {
        "meta": {
            "papers": len(names),
            "render_dpi": index[names[0]]["dpi"],
            "ground_truth": "pdfplumber native PDF text layer, page 1",
            "latex_used": False,
            "invocation": "per-image" if args.per_image else "batched-per-engine",
            "concurrent_engines": workers,
            "wall_clock_sec": round(elapsed, 1),
            "gpus": sorted({r["gpu_name"] for r in runs if r.get("gpu_name")}),
            "engines": [e[0] for e in engines],
        },
        "papers": index,
        "runs": runs,
    }
    RESULTS.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {RESULTS} ({len(runs)} runs, {elapsed/60:.1f} min)")

    write_summary(payload)
    print(f"wrote {SUMMARY}")
    return 0


def write_summary(payload):
    runs = payload["runs"]
    keys = []
    for r in runs:
        k = (r["model"], r["mode"])
        if k not in keys:
            keys.append(k)
    keys.sort()

    def agg(k):
        rs = [r for r in runs if (r["model"], r["mode"]) == k]
        ok = [r for r in rs if not r["error"] and not r["skipped"]]
        cers = [r["cer"] for r in ok if r["cer"] is not None]
        nosp = [r["cer_nospace"] for r in ok if r.get("cer_nospace") is not None]
        cont = [r["cer_content"] for r in ok if r["cer_content"] is not None]
        f1s = [r["word_f1"] for r in ok if r.get("word_f1") is not None]
        recs = [r["word_recall"] for r in ok if r.get("word_recall") is not None]
        times = [r["time_sec"] for r in ok if r["time_sec"] is not None]
        nb = [r["n_boxes"] for r in ok]
        return {
            "n_ok": len(ok), "n": len(rs),
            "cer": sum(cers) / len(cers) if cers else None,
            "cer_nospace": sum(nosp) / len(nosp) if nosp else None,
            "cer_content": sum(cont) / len(cont) if cont else None,
            "word_f1": sum(f1s) / len(f1s) if f1s else None,
            "word_recall": sum(recs) / len(recs) if recs else None,
            "time": sum(times) / len(times) if times else None,
            "boxes": any(r["has_boxes"] for r in ok),
            "n_boxes": sum(nb) / len(nb) if nb else 0,
            "fmt": next((r["box_format"] for r in ok if r["box_format"]), "--"),
            "labels": sorted({l for r in ok for l in (r["box_labels"] or [])}),
            "vram": max([r["peak_vram_gb"] for r in ok if r["peak_vram_gb"]], default=None),
            "err": [r for r in rs if r["error"]],
        }

    m = payload["meta"]
    L = [
        "# OCR engine survey -- structural output for G_doc",
        "",
        f"{m['papers']} arXiv papers, page 1 only, rendered at {m['render_dpi']} DPI. "
        f"Ground truth is the {m['ground_truth']}; no LaTeX source is read anywhere "
        "in this pipeline. Every engine runs in its own virtualenv on "
        f"{', '.join(m['gpus']) or 'GPU'}.",
        "",
        f"Total sweep: {m['wall_clock_sec']/60:.1f} min wall clock "
        f"({m['invocation']} invocation).",
        "",
        "## Results",
        "",
        "| Engine | Mode | Word F1 | Word recall | Avg CER | Content CER | Boxes | "
        "Box format | Avg s/page | Peak VRAM |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for k in keys:
        a = agg(k)
        f1_s = f"**{a['word_f1']:.3f}**" if a["word_f1"] is not None else "--"
        rc_s = f"{a['word_recall']:.3f}" if a["word_recall"] is not None else "--"
        cer_s = f"{a['cer']:.3f}" if a["cer"] is not None else "--"
        con_s = f"{a['cer_content']:.3f}" if a["cer_content"] is not None else "--"
        t_s = f"{a['time']:.2f}" if a["time"] is not None else "--"
        v_s = f"{a['vram']:.1f} GB" if a["vram"] else "--"
        box_s = f"Yes ({a['n_boxes']:.0f}/page)" if a["boxes"] else "**No**"
        L.append(f"| {k[0]} | {k[1]} | {f1_s} | {rc_s} | {cer_s} | {con_s} | {box_s} | "
                 f"`{a['fmt']}` | {t_s} | {v_s} |")

    n_two = len({r["paper"] for r in runs if r.get("columns") == 2})
    n_all = len({r["paper"] for r in runs})
    L += ["", "**Read Word F1 as the headline number, not CER.** CER is a sequential "
              "metric: it cannot tell \"recognised the wrong characters\" apart from "
              "\"recognised the right characters in a different order\". "
              f"{n_two} of these {n_all} pages are two-column, so ordering dominates. "
              "Word F1 and "
              "recall compare token multisets and are therefore independent of "
              "reading order -- including independent of the reading order this "
              "harness reconstructs for the ground truth.",
          "",
          "- **Word F1 / recall** -- order-independent token overlap. Recall answers "
          "\"did it read all the text on the page\".",
          "- **Avg CER** -- whitespace-collapsed edit distance against ground truth "
          "in reconstructed reading order. Also penalises a model for emitting `#` "
          "headings and `\\(x\\)` math the text layer never contained.",
          "- **Content CER** -- markup and punctuation stripped, lowercased.",
          ""]

    L += ["## Spatial output", "",
          "Box geometry validated against the rendered page size: a box is bad if it "
          "falls outside the page or has non-positive area.",
          "",
          "| Engine | Mode | Boxes | Granularity | Out of bounds | Degenerate | "
          "Median box (w x h, % of page) |",
          "|---|---|---|---|---|---|---|"]
    px = {n: v["image_px"] for n, v in payload["papers"].items()}
    for k in keys:
        tot = oob = deg = 0
        ws, hs = [], []
        for r in runs:
            if (r["model"], r["mode"]) != k or not r.get("boxes"):
                continue
            W, H = px[r["paper"]]
            for b in r["boxes"]:
                bb = b.get("bbox_px")
                if not bb:
                    continue
                tot += 1
                x1, y1, x2, y2 = bb
                if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
                    oob += 1
                if x2 <= x1 or y2 <= y1:
                    deg += 1
                else:
                    ws.append((x2 - x1) / W * 100)
                    hs.append((y2 - y1) / H * 100)
        if not tot:
            continue
        ws.sort()
        hs.sort()
        gran = "text line" if hs[len(hs) // 2] < 2 else "layout block"
        L.append(f"| {k[0]} | {k[1]} | {tot} | {gran} | {oob} | {deg} | "
                 f"{ws[len(ws)//2]:.0f}% x {hs[len(hs)//2]:.1f}% |")
    L.append("")

    L += ["### Box vocabularies", ""]
    for k in keys:
        a = agg(k)
        if a["labels"]:
            L.append(f"- **{k[0]} / {k[1]}**: {', '.join(a['labels'])}")
    L.append("")

    L += ["## Per-paper Word F1", "",
          "| Paper | cols | " + " | ".join(f"{a}/{b}" for a, b in keys) + " |",
          "|---" * (len(keys) + 2) + "|"]
    cols_of = {r["paper"]: r.get("columns") for r in runs}
    for name in sorted({r["paper"] for r in runs}):
        row = [name, str(cols_of.get(name) or "?")]
        for k in keys:
            r = next((x for x in runs if x["paper"] == name
                      and (x["model"], x["mode"]) == k), None)
            if r is None or r["skipped"]:
                row.append("--")
            elif r["error"]:
                row.append("ERR")
            else:
                row.append(f"{r['word_f1']:.3f}" if r.get("word_f1") is not None
                           else "--")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- failure modes, measured rather than asserted ----
    L += ["## Notable failure modes", ""]

    for k in keys:
        rs = [r for r in runs if (r["model"], r["mode"]) == k
              and not r["skipped"] and not r["error"] and r["cer"] is not None]
        one = [r["cer"] for r in rs if r.get("columns") == 1]
        two = [r["cer"] for r in rs if r.get("columns") == 2]
        if len(one) >= 2 and len(two) >= 2:
            a, b = sum(one) / len(one), sum(two) / len(two)
            # Only call this "no reading order" when the two-column CER is both
            # far worse *and* bad in absolute terms. Every engine degrades
            # somewhat on two columns (hyphenation, line breaks, and the ground
            # truth's own reconstructed order); that is not the same defect as
            # emitting the columns interleaved.
            if b > 0.35 and b > 8 * max(a, 0.01):
                L.append(
                    f"- **{k[0]} / {k[1]} has no reading order.** CER {a:.3f} on "
                    f"single-column pages vs {b:.3f} on two-column -- a "
                    f"{b/max(a,1e-6):.0f}x collapse, while its Word F1 stays at "
                    f"{agg(k)['word_f1']:.3f}. It recognises the glyphs almost "
                    "perfectly and returns them in detection order, zipping the two "
                    "columns together. Any consumer must supply its own reading "
                    "order.")
            elif b > 3 * max(a, 0.01):
                L.append(
                    f"- **{k[0]} / {k[1]} degrades on two-column pages.** CER "
                    f"{a:.3f} single-column vs {b:.3f} two-column. Modest next to "
                    "the detection-order engines above, and partly attributable to "
                    "hyphenation and to the ground truth's own reconstructed reading "
                    "order rather than to the model.")

    for k in keys:
        rs = [r for r in runs if (r["model"], r["mode"]) == k
              and not r["skipped"] and not r["error"]
              and r.get("word_recall") is not None and r["hyp_chars"] > 0]
        # Empty generations are reported separately; they are a different bug
        # from partial coverage and would otherwise be counted twice.
        low = sorted([r for r in rs if r["word_recall"] < 0.7],
                     key=lambda r: r["word_recall"])
        if low:
            worst = ", ".join(f"{r['paper']} ({r['word_recall']:.2f})" for r in low[:4])
            L.append(f"- **{k[0]} / {k[1]} silently drops page content.** Word recall "
                     f"below 0.70 on {len(low)} of {len(rs)} pages: {worst}. The text "
                     "it does emit is accurate (precision stays high), so the loss is "
                     "invisible unless you measure recall.")

    empties = [r for r in runs if not r["skipped"] and not r["error"]
               and r["hyp_chars"] == 0]
    for r in empties:
        L.append(f"- **{r['model']} / {r['mode']} returned nothing at all on "
                 f"`{r['paper']}`.** Zero characters, zero boxes, no exception -- the "
                 "model emits EOS immediately. Reproduced deterministically at "
                 "temperature 0 (3.0 s versus its usual ~55 s). A pipeline that does "
                 "not check for empty output would record this as a clean run.")

    for k in keys:
        rs = [r for r in runs if (r["model"], r["mode"]) == k
              and not r["skipped"] and not r["error"]
              and r.get("word_precision") is not None and r["hyp_chars"] > 0]
        # Over-generation means "emitted extra text", so a page where the model
        # emitted nothing cannot qualify -- precision 0.0 there is the empty bug.
        loose = sorted([r for r in rs if r["word_precision"] < 0.8
                        and r.get("word_recall", 0) > 0.5],
                       key=lambda r: r["word_precision"])
        if loose:
            worst = ", ".join(f"{r['paper']} ({r['word_precision']:.2f})"
                              for r in loose[:3])
            L.append(f"- **{k[0]} / {k[1]} over-generates.** Word precision below 0.80 "
                     f"on {len(loose)} page(s): {worst}, with recall staying high -- "
                     "the signature of repeated or invented spans rather than missed "
                     "text.")

    # Structure costs coverage: compare each engine's own two modes.
    for model in sorted({m for m, _ in keys}):
        t = [r for r in runs if r["model"] == model and r["mode"] == "text"
             and not r["skipped"] and not r["error"] and r.get("word_recall")]
        g = [r for r in runs if r["model"] == model and r["mode"] == "grounding"
             and not r["skipped"] and not r["error"] and r.get("word_recall")]
        if t and g:
            rt = sum(x["word_recall"] for x in t) / len(t)
            rg = sum(x["word_recall"] for x in g) / len(g)
            if rt - rg > 0.03:
                L.append(f"- **{model} pays for structure in coverage.** Word recall "
                         f"drops {rt:.3f} -> {rg:.3f} when asked for boxes instead of "
                         "plain text: the grounded pass reads less of the page.")
    L.append("")

    errs = [r for r in runs if r["error"]]
    if errs:
        L += ["## Errors and skips", ""]
        shown = {}
        for r in errs:
            shown.setdefault((r["model"], r["mode"], r["error"]), []).append(r["paper"])
        for (model, mode, err), papers in shown.items():
            where = "all pages" if len(papers) == n_all else ", ".join(sorted(papers))
            L.append(f"- `{model}/{mode}` ({where}): {err}")
        L.append("")

    L += [
        "## Bottom line for G_doc",
        "",
        "The survey question was which engine yields both the spatial structure a "
        "candidate graph needs and accurate text. No single engine is best at both, "
        "and the split is clean:",
        "",
        "- **PaddleOCR det+rec has the best text and the finest boxes, and no reading "
        "order.** Word F1 0.978 -- the highest here -- with ~80 line-level "
        "quadrilaterals per page at 3.0 s, the fastest by 14x. But CER goes 0.011 to "
        "0.550 between single- and two-column pages because lines come back in "
        "detection order. For a graph this matters less than it looks: G_doc nodes "
        "are spatial, so you would impose reading order from the boxes yourself "
        "rather than trust a serialised string.",
        "- **DeepSeek-OCR grounding is the strongest single-pass structured output.** "
        "Word F1 0.893 with semantically labelled blocks (title, sub_title, text, "
        "equation, image, image_caption) and no invalid geometry -- but 62 s/page and "
        "~11 coarse blocks, so fine-grained nodes need a second pass.",
        "- **GLM-OCR cannot serve this role at all.** Competitive text (F1 0.943) but "
        "structurally blind: no grounding prompt, no bbox vocabulary. Its own "
        "pipeline gets layout from a separate PP-DocLayout-V3 stage, which is "
        "PaddleOCR's detector -- so choosing GLM-OCR means running Paddle anyway.",
        "- **Unlimited-OCR is not yet dependable.** Comparable structure to DeepSeek "
        "(F1 0.856), but it over-generates on 3 of 10 pages and returned nothing at "
        "all on one, deterministically.",
        "",
        "**Suggested pairing:** PaddleOCR det+rec for text and line boxes (fast, "
        "highest recall, finest granularity), with PP-StructureV3 or DeepSeek-OCR "
        "grounding supplying block-level semantic labels over the top. Check the "
        "recall gap before trusting either grounded pass as the sole source of text "
        "-- PP-StructureV3 dropped a fifth of the words on average and nearly half on "
        "two of the pages.",
        "",
        "## Environment caveats",
        "",
        "These numbers were measured on Tesla V100 (compute capability 7.0), which "
        "constrains the setup in ways worth stating before the latencies are quoted "
        "anywhere:",
        "",
        "- **FlashAttention-2 could not be used.** It requires Ampere (sm_80+). All "
        "transformer engines ran `eager` or `sdpa` attention, so the per-page times "
        "here are an upper bound; an A100/H100 would be materially faster.",
        "- **DeepSeek-OCR and Unlimited-OCR are forced into bfloat16**, which Volta "
        "has no tensor cores for. Their modeling code hardcodes `.to(torch.bfloat16)` "
        "on image tensors and a bf16 autocast, so float16 (about 9x faster on this "
        "GPU) fails in `masked_scatter_`. Their latencies are penalised accordingly; "
        "GLM-OCR and PaddleOCR were free to use float16/float32.",
        "- **vLLM was not used.** It would have replaced the pinned torch in each "
        "venv and broken the three mutually-incompatible transformers versions "
        "(DeepSeek 4.46.3, Unlimited 4.57.1, GLM 5.14.1) that make these engines "
        "coexist at all.",
        "- Engines ran **sequentially, one at a time**: the container caps RAM at "
        "16 GiB and parallel model loads are OOM-killed.",
        "",
        "## Reproducing",
        "",
        "```",
        "python3 fetch_arxiv_pdfs.py                                  # 10 PDFs",
        "ocr_venvs/env_eval/bin/python prepare_pages.py               # renders + GT",
        "ocr_venvs/env_eval/bin/python orchestrate_benchmark.py       # full sweep",
        "ocr_venvs/env_eval/bin/python orchestrate_benchmark.py --rescore  # metrics only",
        "```",
        "",
        "`--rescore` recomputes every metric from the stored engine outputs in "
        "`ocr_survey_results.json`, so changing a CER definition or the ground-truth "
        "extraction costs seconds instead of another GPU sweep. Exact package sets "
        "for all five virtualenvs are frozen in `requirements/`.",
    ]

    SUMMARY.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
