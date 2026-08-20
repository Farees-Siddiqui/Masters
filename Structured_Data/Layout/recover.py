"""Recover a document's structure as XML, using LayoutLMv3 only.

    python Layout/recover.py -i <document.pdf> -o <output.xml>

Pipeline:

    1. LayoutLMv3 (FUNSD) tags every word QUESTION / ANSWER / HEADER / OTHER
    2. consecutive same-label words on a line merge into chunks
    3. each ANSWER is linked to the nearest QUESTION -- to its left on the same
       row, or directly above it -- by greedy global nearest-first assignment
    4. each linked pair becomes an element named after the question text

Nothing is invented. Specifically:

  * element names are slugified from the label printed on the page, so a
    document saying "Surname" yields <surname>, not <lastname>. No normalising
    table maps surface wording onto a canonical schema.
  * a value the model tagged ANSWER but could not be linked to any key is
    emitted as <unkeyed>, not given a guessed name and not silently dropped.
  * the root is <document>. Nothing on the page says "student-record".
  * if the model finds nothing, the output is an empty <document/>. There is no
    fallback that fills in expected fields.

Grouping into subtrees is NOT attempted by default. FUNSD's HEADER class marks
column headers as often as section headings, so building nesting from it would
manufacture structure the page does not show. --group-by-header turns it on to
see what it does.
"""

import argparse
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_layoutlmv3 import MODEL, classify, runs  # noqa: E402

ROW_MAX_GAP = 250.0   # pt, label to the left of its value
BELOW_MAX_GAP = 40.0  # pt, column header above its value


def slug(text):
    """Element name from the label as printed. Returns None if nothing survives."""
    s = re.sub(r"[^0-9a-z]+", "_", text.lower()).strip("_")
    if not s:
        return None
    return f"f_{s}" if s[0].isdigit() else s


def overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    span = min(a1 - a0, b1 - b0)
    return (hi - lo) / span if span > 0 and hi > lo else 0.0


def candidate_links(questions, answers):
    """(distance, answer_index, question_index) for every plausible pairing."""
    out = []
    for ai, (_, ar, _) in enumerate(answers):
        for qi, (_, qr, _) in enumerate(questions):
            # label to the left, same row
            if overlap(ar.y0, ar.y1, qr.y0, qr.y1) > 0.5 and qr.x1 <= ar.x0 + 2:
                gap = ar.x0 - qr.x1
                if gap <= ROW_MAX_GAP:
                    out.append((gap, ai, qi))
            # column header directly above
            if overlap(ar.x0, ar.x1, qr.x0, qr.x1) > 0.3 and qr.y1 <= ar.y0 + 2:
                gap = ar.y0 - qr.y1
                if gap <= BELOW_MAX_GAP:
                    out.append((gap, ai, qi))
    return sorted(out)


def link(questions, answers):
    """Greedy nearest-first, each key and each value used at most once."""
    pairs, used_q, used_a = {}, set(), set()
    for _, ai, qi in candidate_links(questions, answers):
        if ai in used_a or qi in used_q:
            continue
        pairs[ai] = qi
        used_a.add(ai)
        used_q.add(qi)
    return pairs


FILLER = {"of", "is", "was", "the", "a", "an", "for", "at", "in"}
WEB = re.compile(r"\S+@\S+|www\.\S+|\S+\.(?:ca|com|org|net)\b")


def dehyphenate(text):
    """Undo TeX's line-break hyphenation: 'Satis-\\nfactory' -> 'Satisfactory'."""
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)


def strip_label_prefix(value, label):
    """Drop the key when it leaks into the value span ('grade of 90' -> '90')."""
    if not value.lower().startswith(label.lower()):
        return value
    rest = value[len(label):].lstrip(" :—-")
    while True:
        first, _, tail = rest.partition(" ")
        if first.lower() in FILLER and tail:
            rest = tail.lstrip()
        else:
            break
    return rest or value


def gliner_pass(text, taken, labels, threshold, model):
    """Zero-shot NER over the page text, using the corpus's own field names.

    Only spans the layout pass did not already account for are returned, so
    GLiNER supplements the labelled surfaces rather than competing with them.

    Three kinds of span are rejected rather than repaired:
      * a value identical to its own key ("Evaluation" tagged Evaluation)
      * a value that exists only inside an email address or URL, which is how
        an addressee's domain gets read as a city
      * anything the layout pass already found
    """
    web_tokens = WEB.findall(text)
    found, seen = [], set()
    for ent in model.predict_entities(text, labels, threshold=threshold):
        val = strip_label_prefix(" ".join(ent["text"].split()).strip(" .,"),
                                 ent["label"])
        low = val.lower()
        if not val or low in taken or low == ent["label"].lower():
            continue
        in_web = any(low in t.lower() for t in web_tokens)
        standalone = re.search(rf"(?<![\w@.-]){re.escape(val)}(?![\w@.-])", text)
        if in_web and not standalone:
            continue
        key = (ent["label"].lower(), low)
        if key in seen:
            continue
        seen.add(key)
        found.append((ent["label"], val, ent["score"]))
    return found


def build_xml(source, chunks, group_by_header=False, extra=()):
    questions = [c for c in chunks if c[2] == "QUESTION"]
    answers = [c for c in chunks if c[2] == "ANSWER"]
    headers = [c for c in chunks if c[2] == "HEADER"]
    pairs = link(questions, answers)

    root = ET.Element("document", {"source": source, "model": MODEL})

    def parent_for(rect):
        if not group_by_header:
            return root
        above = [h for h in headers if h[1].y1 <= rect.y0 + 2]
        if not above:
            return root
        h = max(above, key=lambda h: h[1].y1)
        name = slug(h[0])
        if not name:
            return root
        for child in root:
            if child.tag == name and child.get("role") == "header":
                return child
        return ET.SubElement(root, name, {"role": "header", "label": h[0]})

    for ai, (atext, arect, _) in enumerate(answers):
        holder = parent_for(arect)
        if ai in pairs:
            qtext = questions[pairs[ai]][0]
            name = slug(qtext)
            if name:
                el = ET.SubElement(holder, name,
                                   {"label": qtext.strip(), "by": "layoutlmv3"})
            else:
                el = ET.SubElement(holder, "unkeyed", {"by": "layoutlmv3"})
        else:
            el = ET.SubElement(holder, "unkeyed", {"by": "layoutlmv3"})
        el.text = atext.strip()

    for label, value, score in extra:
        name = slug(label)
        attrs = {"label": label, "by": "gliner", "score": f"{score:.2f}"}
        el = ET.SubElement(root, name or "unkeyed", attrs)
        el.text = value

    stats = {
        "questions": len(questions),
        "answers": len(answers),
        "linked": len(pairs),
        "unkeyed": len(answers) - len(pairs),
        "gliner": len(extra),
    }
    return root, stats


def indent(el, level=0):
    pad = "\n" + "  " * level
    if len(el):
        if not (el.text or "").strip():
            el.text = pad + "  "
        for child in el:
            indent(child, level + 1)
        if not (el.tail or "").strip():
            el.tail = pad
        if not (el[-1].tail or "").strip():
            el[-1].tail = pad
    elif level and not (el.tail or "").strip():
        el.tail = pad


def recover(pdf_path, processor, model, device, group_by_header=False,
            gliner=None, labels=None, threshold=0.5):
    doc = fitz.open(pdf_path)
    page = doc[0]
    chunks = runs(classify(page, processor, model, device))
    text = dehyphenate(page.get_text())
    doc.close()

    extra = ()
    if gliner is not None and labels:
        taken = {
            " ".join(c[0].split()).strip(" .,").lower()
            for c in chunks if c[2] == "ANSWER"
        }
        extra = gliner_pass(text, taken, labels, threshold, gliner)
    return build_xml(Path(pdf_path).name, chunks, group_by_header, extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True, help="input PDF")
    ap.add_argument("-o", "--output", required=True,
                    help="output .xml file, or a directory")
    ap.add_argument("--group-by-header", action="store_true",
                    help="nest pairs under the nearest HEADER above them")
    ap.add_argument("--gliner", action="store_true",
                    help="second pass: zero-shot NER over the page text, typed "
                         "with the field names harvested from the corpus")
    ap.add_argument("--labels", default=str(Path(__file__).parent / "label_vocab.json"),
                    help="label_vocab.json from harvest_labels.py")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="minimum GLiNER confidence (default 0.5)")
    ap.add_argument("--gliner-model", default="urchade/gliner_medium-v2.1")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        sys.exit(f"no such file: {src}")
    # a directory processes every PDF in it, so the models load once
    sources = sorted(src.glob("*.pdf")) if src.is_dir() else [src]
    if not sources:
        sys.exit(f"no PDFs in {src}")
    dst_arg = Path(args.output)
    into_dir = len(sources) > 1 or dst_arg.is_dir() or args.output.endswith("/")
    if into_dir:
        dst_arg.mkdir(parents=True, exist_ok=True)
    else:
        dst_arg.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(MODEL, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL).to(device).eval()

    gliner = labels = None
    if args.gliner:
        import json

        from gliner import GLiNER
        labels = json.loads(Path(args.labels).read_text())["transferable"]
        gliner = GLiNER.from_pretrained(args.gliner_model).to(device).eval()

    for one in sources:
        dst = (dst_arg / (one.stem + ".xml")) if into_dir else dst_arg
        root, stats = recover(one, processor, model, device, args.group_by_header,
                              gliner, labels, args.threshold)
        indent(root)
        ET.ElementTree(root).write(dst, encoding="utf-8", xml_declaration=True)
        print(f"{one.name} -> {dst}  "
              f"(keys {stats['questions']}, values {stats['answers']}, "
              f"linked {stats['linked']}, unkeyed {stats['unkeyed']}, "
              f"gliner {stats['gliner']})")


if __name__ == "__main__":
    main()
