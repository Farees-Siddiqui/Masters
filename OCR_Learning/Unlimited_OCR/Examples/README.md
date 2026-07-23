# Unlimited-OCR Examples

Runnable examples for [Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) plus a
harness for comparing it against other OCR engines on the same input.
Parameter choices (prompts, gundam/base mode, repetition guards) follow
`../docs/unlimited-ocr.html` — read that first.

## Files

| File | What it does |
|---|---|
| `00_make_sample.py` | Renders a synthetic document image + ground-truth text into `images/`, so every example runs without hunting for test data |
| `01_unlimited_single_image.py` | Single image through Unlimited-OCR in **gundam** mode |
| `02_unlimited_pdf.py` | PDF → 300 DPI pages (PyMuPDF) → `infer_multi` in **base** mode, with page chunking to stay under `max_length` |
| `ocr_engines.py` | Common adapter interface: `unlimited`, `deepseek`, `paddle`, `tesseract`, `easyocr` — each takes an image path, returns text (`deepseek` lives in `../../DeepSeek_OCR/Examples`) |
| `03_compare_models.py` | Runs the same image through selected engines; reports timing, optional CER/WER against ground truth, and pairwise output similarity |

## Quick start

```bash
pip install -r requirements.txt   # see comments inside — engines are optional

python 00_make_sample.py                          # creates images/sample.png + sample_truth.txt
python 01_unlimited_single_image.py images/sample.png
python 03_compare_models.py images/sample.png --engines unlimited,paddle,tesseract --truth images/sample_truth.txt
```

Engines that aren't installed are skipped with a message, not an error, so you can
compare whatever subset you have. Outputs land in `out/<engine>.txt`.

## Notes

- Unlimited-OCR needs a 12 GB+ GPU (3B params in bf16). The other engines run anywhere.
- Unlimited-OCR emits parsed markdown, the others emit raw text lines — the comparison
  normalizes both (strips markup, collapses whitespace) before computing metrics.
- Unlimited-OCR gives **no bounding boxes**; if you need spatial grounding, pair it
  with a layout detector (see the survival guide, §6).
