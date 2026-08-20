#!/usr/bin/env python3
"""GLM-OCR runner -- env_glm, GPU only.

  ocr_venvs/env_glm/bin/python run_glm.py --image page.png [--mode text|grounding]

GLM-OCR is a 0.9B recogniser with a deliberately closed prompt set. Its model
card allows exactly two prompt families: document parsing ("Text Recognition:",
"Formula Recognition:", "Table Recognition:") and JSON-schema information
extraction. There is no grounding/detection prompt and no bbox vocabulary -- in
the official pipeline the layout stage is a *separate* model (PP-DocLayout-V3)
whose boxes are fed to GLM-OCR as crops.

So --mode grounding does not fall back to something that looks like boxes; it
reports unsupported, which is the honest answer for a G_doc feasibility survey.
"""
import argparse
import json
import sys
import time

MODEL_ID = "zai-org/GLM-OCR"
TEXT_PROMPT = "Text Recognition:"


def unsupported(image, note):
    return {
        "model": "glm-ocr",
        "mode": "grounding",
        "image": image,
        "text": "",
        "has_boxes": False,
        "boxes": None,
        "box_format": None,
        "time_sec": 0.0,
        "device": "cuda:0",
        "skipped": True,
        "error": note,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True)
    ap.add_argument("--mode", choices=["text", "grounding"], default="text")
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    args = ap.parse_args()

    note = ("GLM-OCR exposes no grounding prompt and no bbox tokens; layout boxes "
            "come from a separate PP-DocLayout-V3 stage in the official pipeline.")
    if args.mode == "grounding":
        out = [unsupported(i, note) for i in args.image]
        json.dump(out[0] if len(out) == 1 else out, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return

    import torch
    assert torch.cuda.is_available(), "CUDA unavailable -- refusing to run on CPU"
    device = "cuda:0"
    torch.cuda.set_device(0)
    cap = torch.cuda.get_device_capability(0)

    from transformers import AutoModelForImageTextToText, AutoProcessor

    attn = "flash_attention_2" if cap[0] >= 8 else "sdpa"
    # Native weights are bf16, but Volta has no bf16 tensor cores and this model
    # (unlike the DeepSeek pair) does not hardcode a bf16 autocast, so fp16 is
    # both safe and roughly 9x faster here.
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation=attn,
    ).to(device).eval()
    load_sec = time.time() - t0

    real_dev = str(next(model.parameters()).device)
    assert real_dev.startswith("cuda"), f"model landed on {real_dev}"

    results = []
    for image in args.image:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        t1 = time.time()
        try:
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            inputs.pop("token_type_ids", None)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            raw = processor.decode(gen[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            err = None
        except Exception as exc:  # noqa: BLE001
            raw, err = "", f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        elapsed = time.time() - t1

        results.append({
            "model": "glm-ocr",
            "mode": "text",
            "image": image,
            "text": (raw or "").strip(),
            "has_boxes": False,
            "boxes": None,
            "box_format": None,
            "time_sec": round(elapsed, 3),
            "device": real_dev,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{cap[0]}.{cap[1]}",
            "dtype": str(dtype).replace("torch.", ""),
            "attn_impl": attn,
            "load_sec": round(load_sec, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated(0) / 1e9, 2),
            "raw_chars": len(raw or ""),
            "box_support_note": note,
            "error": err,
        })

    json.dump(results[0] if len(results) == 1 else results, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
