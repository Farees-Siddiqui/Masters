#!/usr/bin/env python3
"""Unlimited-OCR runner -- env_unlimited, GPU only.

  ocr_venvs/env_unlimited/bin/python run_unlimited.py --image page.png [--mode text|grounding]

Same JSON contract as the other runners. Unlimited-OCR is architecturally a
DeepSeek-OCR derivative (same deepencoder, same <|grounding|> vocabulary, same
infer() signature), so the box handling is identical -- but it needs its own venv
because it pins transformers 4.57.1 against DeepSeek's 4.46.3.
"""
import argparse
import contextlib
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import time

MODEL_ID = "baidu/Unlimited-OCR"

PROMPTS = {
    "text": "<image>\nFree OCR. ",
    "grounding": "<image>\n<|grounding|>Convert the document to markdown. ",
}

# Unlimited-OCR shares DeepSeek's grounding vocabulary but arranges it its own
# way. What it actually emits, verified against a dumped raw generation, is:
#     <|det|>aside_text [23, 256, 61, 708]<|/det|>arXiv:1505.04597v1 ...
# i.e. label and box together inside <|det|>, with the block's text following the
# closing tag and running until the next <|det|>. Coordinates are on a 0-999
# grid, same as DeepSeek.
DET_RE = re.compile(
    r"<\|det\|>\s*([A-Za-z][A-Za-z0-9_]*)?\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
    r"\s*(\d+)\s*\]\s*<\|/det\|>"
)
# DeepSeek-style <|ref|>label<|/ref|><|det|>[[...]]<|/det|>, kept as a fallback.
REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.DOTALL)
TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def _scale(x1, y1, x2, y2, width, height):
    return [round(x1 / 999 * width, 1), round(y1 / 999 * height, 1),
            round(x2 / 999 * width, 1), round(y2 / 999 * height, 1)]


def parse_grounding(raw, width, height):
    marks = list(DET_RE.finditer(raw))
    if marks:
        boxes, texts = [], []
        for i, m in enumerate(marks):
            x1, y1, x2, y2 = (int(m.group(k)) for k in range(2, 6))
            end = marks[i + 1].start() if i + 1 < len(marks) else len(raw)
            body = TOKEN_RE.sub("", raw[m.end():end]).strip()
            boxes.append({
                "label": m.group(1) or "block",
                "text": body,
                "bbox_px": _scale(x1, y1, x2, y2, width, height),
                "bbox_norm_1000": [x1, y1, x2, y2],
            })
            texts.append(body)
        return "\n\n".join(t for t in texts if t), boxes

    boxes = []
    for label, coord_blob in REF_RE.findall(raw):
        try:
            coords = json.loads(coord_blob)
        except json.JSONDecodeError:
            continue
        for c in coords:
            if isinstance(c, list) and len(c) == 4:
                boxes.append({
                    "label": label.strip(),
                    "bbox_px": _scale(c[0], c[1], c[2], c[3], width, height),
                    "bbox_norm_1000": c,
                })
    return TOKEN_RE.sub("", REF_RE.sub("", raw)).strip(), boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--mode", choices=sorted(PROMPTS), default="grounding")
    args = ap.parse_args()

    import torch
    assert torch.cuda.is_available(), "CUDA unavailable -- refusing to run on CPU"
    device = "cuda:0"
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_capability(0)

    from transformers import AutoModel, AutoTokenizer

    attn = "flash_attention_2" if cap[0] >= 8 else "eager"
    # Same constraint as DeepSeek-OCR: infer() hardcodes bf16 image tensors and a
    # bf16 autocast, so the LM must be bf16 too regardless of what Volta prefers.
    dtype = torch.bfloat16
    bf16_native = torch.cuda.is_bf16_supported()

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation=attn,
    )
    model = model.eval().to(device=device, dtype=dtype)
    load_sec = time.time() - t0

    real_dev = str(next(model.parameters()).device)
    assert real_dev.startswith("cuda"), f"model landed on {real_dev}"

    from PIL import Image

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for image in args.image:
            width, height = Image.open(image).size
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            t1 = time.time()
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), torch.no_grad():
                    raw = model.infer(
                        tokenizer,
                        prompt=PROMPTS[args.mode],
                        image_file=image,
                        output_path=tmp,
                        base_size=1024,
                        image_size=640,
                        crop_mode=True,
                        save_results=False,
                        eval_mode=True,
                    )
                err = None
            except Exception as exc:  # noqa: BLE001
                raw, err = "", f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()
            elapsed = time.time() - t1

            raw = raw or ""
            if os.environ.get("OCR_DUMP_RAW"):
                pathlib.Path(os.environ["OCR_DUMP_RAW"]).write_text(raw, encoding="utf-8")
            if args.mode == "grounding":
                text, boxes = parse_grounding(raw, width, height)
            else:
                text, boxes = re.sub(r"<\|[^|>]*\|>", "", raw).strip(), []

            results.append({
                "model": "unlimited-ocr",
                "mode": args.mode,
                "image": image,
                "text": text,
                "has_boxes": bool(boxes),
                "boxes": boxes or None,
                "box_format": "xyxy_pixels (from 0-999 normalised grid)" if boxes else None,
                "time_sec": round(elapsed, 3),
                "device": real_dev,
                "gpu_name": torch.cuda.get_device_name(0),
                "compute_capability": f"{cap[0]}.{cap[1]}",
                "dtype": str(dtype).replace("torch.", ""),
                "dtype_forced_by_model_code": True,
                "bf16_native_on_this_gpu": bf16_native,
                "attn_impl": attn,
                "load_sec": round(load_sec, 2),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
                "raw_chars": len(raw),
                "error": err,
            })

    json.dump(results[0] if len(results) == 1 else results, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
