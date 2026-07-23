# DeepSeek-OCR Examples

Runnable examples for [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR),
structured like `../../Unlimited_OCR/Examples`. Parameter choices (prompts,
resolution modes, grounding format) follow `../docs/deepseek-ocr.html` — read
that first. DeepSeek-OCR is the predecessor
of Unlimited-OCR: a ~3B VLM (380M DeepEncoder + 3B MoE decoder, ~570M active)
that parses a page image into markdown in one pass, built around "contexts
optical compression" — a page becomes 64–400 vision tokens depending on mode.

## Files

| File | What it does |
|---|---|
| `deepseek_engine.py` | Loader + `parse_image()` helper: fixed prompts, the five resolution modes, flash-attn → eager fallback for Windows |
| `01_deepseek_single_image.py` | Single image in **gundam** mode (`--prompt free` for plain text) |
| `02_deepseek_pdf.py` | PDF → PyMuPDF pages → per-page `infer` loop; combined `result.md` with `<PAGE>` markers to match the Unlimited-OCR output shape |

For cross-engine comparison use `../../Unlimited_OCR/Examples/03_compare_models.py`
— the `deepseek` engine is registered in its `ocr_engines.py`.

## Quick start

```bash
# Needs its own venv (../.venv) — the pinned transformers 4.46.3 is load-bearing;
# the Unlimited-OCR venv's 4.57.1 fails to load the remote code (LlamaFlashAttention2).
pip install -r requirements.txt   # read the pin + cu126 notes inside first

python 01_deepseek_single_image.py ../../Unlimited_OCR/Examples/images/sample.png
python 02_deepseek_pdf.py ../../../AST/resnet.pdf -o out/resnet
```

First run downloads ~6 GB of weights from the Hub (`deepseek-ai/DeepSeek-OCR`,
`trust_remote_code=True`).

## Key differences from Unlimited-OCR

- **Bounding boxes exist here.** The `<|grounding|>` markdown prompt emits
  `<|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` spans (coords normalized
  to 0–1000). `save_results=True` writes the cleaned markdown to `result.mmd`
  plus a boxed render (`result_with_boxes.jpg`) and figure crops. Unlimited-OCR
  gives no coordinates at all.
- **No `infer_multi`.** Strictly one image per call; multi-page context
  (tables spanning page breaks) is Unlimited-OCR's addition.
- **Five resolution modes** instead of two: tiny (512, 64 tokens), small
  (640, 100), base (1024, 256), large (1280, 400), gundam (1024 global +
  640 crops). Token budget is the compression knob: ~97% decoding precision
  under 10× text-token compression, degrading to ~60% around 20×.
- **Prompts differ**: `"<image>\n<|grounding|>Convert the document to markdown. "`
  or `"<image>\nFree OCR. "` (exact strings, trailing space included) vs
  Unlimited's `"<image>document parsing."`.
- Serving path for throughput is vLLM (the repo ships `run_dpsk_ocr_pdf.py`);
  the Transformers path here is for experiments.
