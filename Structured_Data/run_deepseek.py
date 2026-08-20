#!/usr/bin/env python3
"""DeepSeek-OCR runner -- env_deepseek, GPU only.

  ocr_venvs/env_deepseek/bin/python run_deepseek.py --image page.png [--mode text|grounding]

Emits one JSON object per image on stdout:
  {"text", "has_boxes", "boxes", "time_sec", "device", ...}
Several --image flags emit a JSON array instead, so the orchestrator can pay the
model-load cost once instead of ten times.

Volta notes (V100, sm_70): weights are bfloat16 but Volta has no bf16 tensor
cores, so we load float16. FlashAttention-2 needs Ampere+, so attention is eager.
Both facts are reported in the output rather than assumed.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time

MODEL_ID = "deepseek-ai/DeepSeek-OCR"

PROMPTS = {
    "text": "<image>\nFree OCR. ",
    "grounding": "<image>\n<|grounding|>Convert the document to markdown. ",
}

# DeepSeek's documented form: <|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2],...]<|/det|>
REF_RE = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.DOTALL)
# The sibling Unlimited-OCR form, accepted too so a dialect switch cannot silently
# read as "this model has no boxes": <|det|>label [x1,y1,x2,y2]<|/det|>text
DET_RE = re.compile(
    r"<\|det\|>\s*([A-Za-z][A-Za-z0-9_]*)?\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,"
    r"\s*(\d+)\s*\]\s*<\|/det\|>"
)
TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def _scale(x1, y1, x2, y2, width, height):
    # model space is a 0-999 grid; scale back onto the real page
    return [round(x1 / 999 * width, 1), round(y1 / 999 * height, 1),
            round(x2 / 999 * width, 1), round(y2 / 999 * height, 1)]


def parse_grounding(raw, width, height):
    """Pull boxes out of grounding output and return (clean_text, boxes)."""
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
    if boxes:
        return TOKEN_RE.sub("", REF_RE.sub("", raw)).strip(), boxes

    marks = list(DET_RE.finditer(raw))
    texts = []
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
    if boxes:
        return "\n\n".join(t for t in texts if t), boxes
    return TOKEN_RE.sub("", raw).strip(), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True,
                    help="page image; repeat to batch under one model load")
    ap.add_argument("--mode", choices=sorted(PROMPTS), default="grounding")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    args = ap.parse_args()

    import torch
    # MANDATORY GPU: refuse to produce numbers that silently came from a CPU.
    assert torch.cuda.is_available(), "CUDA unavailable -- refusing to run on CPU"
    device = "cuda:0"
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_capability(0)

    from transformers import AutoModel, AutoTokenizer

    # flash-attn is the documented backend but needs sm_80+; on sm_70 it is absent.
    attn = "flash_attention_2" if cap[0] >= 8 else "eager"
    # dtype is not ours to choose: infer() hardcodes `.to(torch.bfloat16)` on the
    # image tensors and runs the vision tower under a bf16 autocast. Loading fp16
    # (which Volta would much prefer) dies in masked_scatter_ with
    # "expected self and source to have same dtypes but got Half and Float".
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
            # infer() streams to stdout unless eval_mode=True, which returns the
            # string instead -- keep stdout clean for the JSON contract.
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
            except Exception as exc:  # noqa: BLE001 - one bad page must not kill the sweep
                raw, err = "", f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize()
            elapsed = time.time() - t1

            raw = raw or ""
            if args.mode == "grounding":
                text, boxes = parse_grounding(raw, width, height)
            else:
                text, boxes = re.sub(r"<\|[^|>]*\|>", "", raw).strip(), []

            results.append({
                "model": "deepseek-ocr",
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
