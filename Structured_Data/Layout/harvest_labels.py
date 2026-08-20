"""Collect the field names the corpus prints on its own pages.

Runs LayoutLMv3 over every document and keeps every span it tagged QUESTION.
Those are the labels the documents themselves display -- "Last Name", "Surname",
"Municipality", "Numeric Grade" -- and they become the candidate type names
handed to zero-shot NER for the documents that print no labels at all.

The point is that this vocabulary is never authored. It is read off the
structured surfaces and transferred to the unstructured ones.

Writes Layout/label_vocab.json:
    {"labels": {"<label text>": ["doc03", "doc05"]}, ...}
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_layoutlmv3 import MODEL, classify, runs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "StudentRecord" / "Output"
OUT = Path(__file__).resolve().parent / "label_vocab.json"

# a label is a short caption, not a sentence the tagger mislabelled
MAX_WORDS = 4
MIN_CHARS = 3


def clean(text):
    t = re.sub(r"\s+", " ", text).strip().strip(":—-–").strip()
    t = re.sub(r"^SECTION\s+\d+\s*[—–-]\s*", "", t, flags=re.I)
    return t


def usable(text):
    if len(text) < MIN_CHARS or len(text.split()) > MAX_WORDS:
        return False
    # must contain letters, and not look like a sentence fragment
    return bool(re.search(r"[A-Za-z]", text)) and not text.endswith(".")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL).to(device).eval()

    labels = defaultdict(list)
    for pdf in sorted(SRC.glob("doc*.pdf")):
        doc = fitz.open(pdf)
        chunks = runs(classify(doc[0], processor, model, device))
        doc.close()
        found = set()
        for text, _, lab in chunks:
            if lab != "QUESTION":
                continue
            t = clean(text)
            if usable(t):
                found.add(t)
        for t in sorted(found):
            labels[t].append(pdf.stem[:5])
        print(f"{pdf.stem[:5]}: {sorted(found)}")

    # A caption seen in only one document may be a real field name or may be a
    # word the tagger mislabelled ("still", "Dear", "Academic"). One seen in
    # several is corroborated by the corpus rather than by my judgement, so
    # agreement across documents is the filter. Case is folded first, since
    # doc05 prints ADDRESS and doc10 prints Address.
    by_fold = defaultdict(set)
    surface = {}
    for text, docs in labels.items():
        by_fold[text.lower()].update(docs)
        surface.setdefault(text.lower(), text)
    transferable = sorted(
        surface[k] for k, docs in by_fold.items() if len(docs) >= 2
    )

    OUT.write_text(json.dumps({
        "labels": dict(sorted(labels.items())),
        "transferable": transferable,
    }, indent=2))
    print(f"\n{len(labels)} distinct labels, "
          f"{len(transferable)} seen in 2+ documents -> {OUT}")
    print("transferable:", transferable)


if __name__ == "__main__":
    main()
