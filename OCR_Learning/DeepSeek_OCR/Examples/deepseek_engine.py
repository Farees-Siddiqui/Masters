"""Loader and inference helpers for DeepSeek-OCR (deepseek-ai/DeepSeek-OCR).

Prompts, resolution modes, and knobs follow the model card
(huggingface.co/deepseek-ai/DeepSeek-OCR). Unlike Unlimited-OCR there is no
infer_multi: the model is strictly single-image, so PDFs are looped page by
page (see 02_deepseek_pdf.py).

flash-attn is the documented attention backend but rarely builds on Windows;
loading falls back to eager attention automatically.
"""

from __future__ import annotations

from pathlib import Path

# Fixed prompt strings from the model card — copy exactly, trailing space included.
PROMPT_MARKDOWN = "<image>\n<|grounding|>Convert the document to markdown. "
PROMPT_FREE_OCR = "<image>\nFree OCR. "
PROMPT_FIGURE = "<image>\nParse the figure. "

# Resolution modes: vision-token budget grows with resolution. Gundam adds
# cropped local tiles on top of the global view — best for dense pages.
MODES = {
    "tiny": dict(base_size=512, image_size=512, crop_mode=False),  # 64 tokens
    "small": dict(base_size=640, image_size=640, crop_mode=False),  # 100 tokens
    "base": dict(base_size=1024, image_size=1024, crop_mode=False),  # 256 tokens
    "large": dict(base_size=1280, image_size=1280, crop_mode=False),  # 400 tokens
    "gundam": dict(base_size=1024, image_size=640, crop_mode=True),
}

_MODEL = None


def load():
    global _MODEL
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-OCR", trust_remote_code=True
        )
        try:
            model = AutoModel.from_pretrained(
                "deepseek-ai/DeepSeek-OCR",
                trust_remote_code=True,
                use_safetensors=True,
                _attn_implementation="flash_attention_2",
            )
        except Exception as exc:  # noqa: BLE001 - flash-attn missing, typical on Windows
            print(
                f"flash_attention_2 unavailable ({exc.__class__.__name__}); "
                "falling back to eager attention"
            )
            model = AutoModel.from_pretrained(
                "deepseek-ai/DeepSeek-OCR",
                trust_remote_code=True,
                use_safetensors=True,
                _attn_implementation="eager",
            )
        _MODEL = (tokenizer, model.eval().cuda().to(torch.bfloat16))
    return _MODEL


def parse_image(
    image_path: Path,
    out_dir: Path,
    mode: str = "gundam",
    prompt: str = PROMPT_MARKDOWN,
) -> str:
    """OCR one image; returns markdown and saves full results (boxed render,
    figure crops, result.mmd) into out_dir."""
    tokenizer, model = load()
    out_dir.mkdir(parents=True, exist_ok=True)
    result = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=str(image_path),
        output_path=str(out_dir),
        save_results=True,
        **MODES[mode],
    )
    # infer() returns the raw stream (grounding tokens included); result.mmd
    # is the cleaned markdown that save_results writes.
    mmd = out_dir / "result.mmd"
    if mmd.exists():
        return mmd.read_text(encoding="utf-8")
    if isinstance(result, str) and result.strip():
        return result
    raise RuntimeError(f"DeepSeek-OCR produced no readable output in {out_dir}")
