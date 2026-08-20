"""LayoutLMv3 (FUNSD-finetuned) over the student-record PDFs.

Unlike text NER, this model sees each token's position on the page. Its label
set is FUNSD's, not ours -- HEADER / QUESTION / ANSWER / OTHER -- so it does not
name our fields. What it is being asked here is the prior question:

    can it tell which spans on the page are keys and which are values?

That is steps 1-2 of the reversal. Word boxes come from the PDF itself, so
apply_ocr is off and the model gets exact text with true coordinates.

Writes Layout/<doc>.lmv3.pdf and prints a per-document tally.
"""

import re
import sys
from pathlib import Path

import fitz
import torch
from PIL import Image
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "StudentRecord"))
from build_gold import RECORDS  # noqa: E402

MODEL = "nielsr/layoutlmv3-finetuned-funsd"
SRC = ROOT / "StudentRecord" / "Output"
OUT = Path(__file__).resolve().parent
FIELDS = ["lastname", "firstname", "street", "city", "grade", "evaluation"]

COLORS = {
    "QUESTION": (0.20, 0.45, 0.85),   # a key
    "ANSWER":   (0.10, 0.60, 0.30),   # a value
    "HEADER":   (0.60, 0.30, 0.75),
}
HIT, PARTIAL, MISS = (0.10, 0.55, 0.25), (0.95, 0.60, 0.00), (0.85, 0.10, 0.10)


def page_words(page):
    """(text, fitz.Rect, box scaled to 0-1000) for every word on the page."""
    w, h = page.rect.width, page.rect.height
    out = []
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        box = [
            max(0, min(1000, int(1000 * x0 / w))),
            max(0, min(1000, int(1000 * y0 / h))),
            max(0, min(1000, int(1000 * x1 / w))),
            max(0, min(1000, int(1000 * y1 / h))),
        ]
        out.append((word, fitz.Rect(x0, y0, x1, y1), box))
    return out


def classify(page, processor, model, device, max_len=512):
    """Tag every word on the page with a FUNSD label.

    transformers 5.x swapped LayoutLMv3's tokenizer for a generic backend whose
    __call__ has no `boxes` argument, so the processor drops the geometry
    silently. The encoding is therefore built by hand, the way
    LayoutLMv3TokenizerFast used to: each word's box is repeated across its
    sub-tokens, and the CLS/SEP boxes are zeros.
    """
    words = page_words(page)
    if not words:
        return []
    pix = page.get_pixmap(dpi=150)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    tok = processor.tokenizer
    input_ids, bbox, word_of_token = [], [], []
    for i, (text, _, box) in enumerate(words):
        # leading space reproduces add_prefix_space for the RoBERTa BPE
        ids = tok(" " + text, add_special_tokens=False)["input_ids"]
        for tid in ids:
            input_ids.append(tid)
            bbox.append(box)
            word_of_token.append(i)

    limit = max_len - 2
    input_ids, bbox, word_of_token = input_ids[:limit], bbox[:limit], word_of_token[:limit]
    input_ids = [tok.cls_token_id] + input_ids + [tok.sep_token_id]
    bbox = [[0, 0, 0, 0]] + bbox + [[0, 0, 0, 0]]
    word_of_token = [None] + word_of_token + [None]

    pixel_values = processor.image_processor(images=[image], return_tensors="pt")["pixel_values"]
    batch = {
        "input_ids": torch.tensor([input_ids]),
        "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
        "bbox": torch.tensor([bbox]),
        "pixel_values": pixel_values,
    }
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits[0]
    pred = logits.argmax(-1).tolist()

    labels = {}
    for tok_i, w_i in enumerate(word_of_token):
        if w_i is None or w_i in labels:
            continue
        labels[w_i] = model.config.id2label[pred[tok_i]]

    # the B-/I- prefix is kept: it is what separates two adjacent entities,
    # e.g. neighbouring header cells in a table row
    return [
        (text, rect, labels.get(i, "O").upper())
        for i, (text, rect, _) in enumerate(words)
    ]


def base(label):
    return re.sub(r"^[BIES]-", "", label)


def runs(tagged):
    """Merge consecutive same-label words on the same line into one span.

    FUNSD entities are multi-word; labelling every word separately would put a
    hundred tiny captions on the page.
    """
    out = []
    cur_text, cur_rect, cur_lab = None, None, None
    for text, rect, raw in tagged:
        lab = base(raw)
        same_line = (
            cur_rect is not None
            and abs(rect.y0 - cur_rect.y0) < 3
            and rect.x0 - cur_rect.x1 < 25
        )
        # B- opens a new entity even when it abuts one of the same class, but
        # only across a real gap: neighbouring table cells are far apart, while
        # words inside one value are a single space apart
        starts_new = raw.startswith("B-") and (
            cur_rect is None or rect.x0 - cur_rect.x1 > 8.0
        )
        if lab == cur_lab and same_line and not starts_new:
            cur_text += " " + text
            cur_rect = cur_rect | rect
        else:
            if cur_lab in COLORS:
                out.append((cur_text, cur_rect, cur_lab))
            cur_text, cur_rect, cur_lab = text, fitz.Rect(rect), lab
    if cur_lab in COLORS:
        out.append((cur_text, cur_rect, cur_lab))
    return out


def annotate(pdf_path, processor, model, device):
    doc_id = pdf_path.stem[:5]
    gold = dict(zip(FIELDS, RECORDS[doc_id]))
    doc = fitz.open(pdf_path)
    page = doc[0]
    tagged = classify(page, processor, model, device)

    for run_text, run_rect, lab in runs(tagged):
        col = COLORS[lab]
        a = page.add_highlight_annot(run_rect)
        a.set_colors(stroke=col)
        a.set_info(content=f"{lab}: {run_text}")
        a.set_opacity(0.30)
        a.update()
        page.insert_text(fitz.Point(run_rect.x0, max(run_rect.y0 - 1.5, 5)),
                         lab, fontsize=4.5, color=col, fontname="helv")

    # did the model mark the six values as ANSWER?
    tally = {"hit": 0, "partial": 0, "miss": 0}
    for field in FIELDS:
        value = str(gold[field])
        toks = [t for t in re.split(r"\s+", value) if t]
        marked = 0
        for tok in toks:
            for text, _, lab in tagged:
                if tok.strip(",.").lower() == text.strip(",.").lower() and base(lab) == "ANSWER":
                    marked += 1
                    break
        state = "hit" if marked == len(toks) else "partial" if marked else "miss"
        tally[state] += 1
        col = {"hit": HIT, "partial": PARTIAL, "miss": MISS}[state]
        rects = page.search_for(value)
        for r in rects:
            page.draw_rect(r + (-1, -1, 1, 1), color=col, width=0.8)
        if rects:
            suffix = {"hit": "", "partial": " PARTIAL", "miss": " not ANSWER"}[state]
            page.insert_text(fitz.Point(rects[0].x0, max(rects[0].y0 - 1.5, 5)),
                             f"{field}{suffix}", fontsize=4.5, color=col, fontname="helv")

    x, y = 40, page.rect.height - 46
    page.draw_line(fitz.Point(x, y - 10), fitz.Point(page.rect.width - 40, y - 10),
                   color=(0.75, 0.75, 0.75), width=0.5)
    page.insert_text(fitz.Point(x, y), f"LayoutLMv3 / FUNSD  ({MODEL})",
                     fontsize=6, color=(0.3, 0.3, 0.3), fontname="hebo")
    cx = x
    for lab, col in COLORS.items():
        page.draw_rect(fitz.Rect(cx, y + 4, cx + 6, y + 10), color=None, fill=col)
        page.insert_text(fitz.Point(cx + 8, y + 9.5), lab, fontsize=5.5, color=col,
                         fontname="helv")
        cx += 14 + 3.4 * len(lab)
    for dx, col, txt in ((0, HIT, "gold value: tagged ANSWER"),
                         (130, PARTIAL, "partly tagged"),
                         (232, MISS, "not tagged ANSWER")):
        page.draw_rect(fitz.Rect(x + dx, y + 14, x + dx + 6, y + 20), color=col, width=0.9)
        page.insert_text(fitz.Point(x + dx + 8, y + 19.5), txt, fontsize=5.5,
                         color=col, fontname="helv")

    out = OUT / f"{pdf_path.stem}.lmv3.pdf"
    doc.save(out)
    doc.close()

    counts = {}
    for _, _, lab in tagged:
        counts[base(lab)] = counts.get(base(lab), 0) + 1
    return out, counts, tally


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL).to(device).eval()
    print("labels:", model.config.id2label)

    header = (f"{'document':<40}{'Q':>5}{'A':>5}{'H':>5}{'O':>5}"
              f"{'hit':>6}{'part':>6}{'miss':>6}")
    print(header)
    print("-" * len(header))
    totals = {"hit": 0, "partial": 0, "miss": 0}
    for pdf in sorted(SRC.glob("doc*.pdf")):
        out, counts, tally = annotate(pdf, processor, model, device)
        for k in totals:
            totals[k] += tally[k]
        print(f"{out.name:<40}{counts.get('QUESTION', 0):>5}{counts.get('ANSWER', 0):>5}"
              f"{counts.get('HEADER', 0):>5}{counts.get('O', 0):>5}"
              f"{tally['hit']:>6}{tally['partial']:>6}{tally['miss']:>6}")
    print("-" * len(header))
    print(f"{'60 values':<40}{'':>20}{totals['hit']:>6}{totals['partial']:>6}"
          f"{totals['miss']:>6}")


if __name__ == "__main__":
    main()
