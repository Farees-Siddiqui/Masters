#!/usr/bin/env python3
"""PaddleOCR runner -- env_paddle, GPU only.

  ocr_venvs/env_paddle/bin/python run_paddle.py --image page.png [--mode text|grounding]

Two genuinely different pipelines, not two prompts:
  text      -- PaddleOCR det+rec. Per-text-line quadrilaterals (4 corner points).
  grounding -- PP-StructureV3. Layout blocks (title/text/table/figure/formula)
               with axis-aligned boxes plus a markdown serialisation.

Note on the API: paddleocr 3.x has no `use_gpu` flag (that was the 2.x
interface). GPU selection is `device="gpu:0"`, and per the previous run's notes
the process must be pinned with CUDA_VISIBLE_DEVICES so that index 0 really is
the card doing the work -- otherwise the VRAM reading comes off an idle GPU.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


def _tolist(v):
    return v.tolist() if hasattr(v, "tolist") else v


def _quad_to_xyxy(quad):
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]


def run_text(engine, image):
    """det+rec: one box per recognised text line."""
    res = engine.predict(image)
    r = res[0] if isinstance(res, list) else res
    d = r.json.get("res", r.json) if hasattr(r, "json") else dict(r)

    texts = d.get("rec_texts") or []
    polys = _tolist(d.get("rec_polys") if d.get("rec_polys") is not None
                    else d.get("dt_polys")) or []
    scores = _tolist(d.get("rec_scores")) or []

    boxes = []
    for i, t in enumerate(texts):
        quad = _tolist(polys[i]) if i < len(polys) else None
        boxes.append({
            "label": "text_line",
            "text": t,
            "quad": quad,
            "bbox_px": _quad_to_xyxy(quad) if quad else None,
            "score": round(float(scores[i]), 4) if i < len(scores) else None,
        })
    return "\n".join(texts), boxes, "quad_4point_pixels (+ xyxy derived)"


def run_grounding(engine, image):
    """PP-StructureV3: layout blocks with class labels, plus markdown."""
    res = engine.predict(image)
    r = res[0] if isinstance(res, list) else res
    d = r.json.get("res", r.json) if hasattr(r, "json") else dict(r)

    boxes = []
    layout = (d.get("layout_det_res") or {}).get("boxes") or []
    for b in layout:
        coord = _tolist(b.get("coordinate"))
        boxes.append({
            "label": b.get("label"),
            "bbox_px": [round(float(c), 1) for c in coord] if coord else None,
            "score": round(float(b["score"]), 4) if b.get("score") is not None else None,
        })

    # Prefer the markdown serialisation; fall back to concatenated OCR lines.
    text = ""
    md = getattr(r, "markdown", None)
    if isinstance(md, dict):
        text = md.get("markdown_texts") or ""
    elif isinstance(md, str):
        text = md
    if not text:
        ocr = d.get("overall_ocr_res") or {}
        text = "\n".join(ocr.get("rec_texts") or [])
    return text, boxes, "xyxy_pixels (layout blocks)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--mode", choices=["text", "grounding"], default="text")
    ap.add_argument("--device", default="gpu:0")
    args = ap.parse_args()

    import paddle
    # MANDATORY GPU.
    assert paddle.device.is_compiled_with_cuda(), "paddle build has no CUDA"
    assert paddle.device.cuda.device_count() > 0, "no CUDA devices visible"
    assert args.device.startswith("gpu"), f"refusing non-GPU device {args.device}"
    paddle.device.set_device(args.device)
    real_dev = str(paddle.device.get_device())
    assert real_dev.startswith("gpu"), f"paddle is on {real_dev}"
    cap = paddle.device.cuda.get_device_capability(0)

    from paddleocr import PaddleOCR, PPStructureV3

    t0 = time.time()
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        if args.mode == "text":
            engine = PaddleOCR(
                device=args.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        else:
            engine = PPStructureV3(device=args.device)
    load_sec = time.time() - t0

    results = []
    for image in args.image:
        paddle.device.cuda.synchronize()
        t1 = time.time()
        try:
            with contextlib.redirect_stdout(quiet):
                if args.mode == "text":
                    text, boxes, fmt = run_text(engine, image)
                else:
                    text, boxes, fmt = run_grounding(engine, image)
            err = None
        except Exception as exc:  # noqa: BLE001
            text, boxes, fmt, err = "", [], None, f"{type(exc).__name__}: {exc}"
        paddle.device.cuda.synchronize()
        elapsed = time.time() - t1

        results.append({
            "model": "paddleocr",
            "mode": args.mode,
            "pipeline": "PaddleOCR det+rec" if args.mode == "text" else "PP-StructureV3",
            "image": image,
            "text": text,
            "has_boxes": bool(boxes),
            "boxes": boxes or None,
            "box_format": fmt if boxes else None,
            "time_sec": round(elapsed, 3),
            "device": real_dev,
            "gpu_name": paddle.device.cuda.get_device_name(0),
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "dtype": "float32",
            "attn_impl": "n/a (CNN detector + CRNN recogniser)",
            "load_sec": round(load_sec, 2),
            "peak_vram_gb": round(paddle.device.cuda.max_memory_allocated(0) / 1e9, 2),
            "raw_chars": len(text),
            "error": err,
        })

    json.dump(results[0] if len(results) == 1 else results, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
