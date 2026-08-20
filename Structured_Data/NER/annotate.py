"""Draw what off-the-shelf NER sees onto the student-record PDFs.

Two layers per page:

  filled highlight   an entity spaCy returned, coloured by its label, with the
                     label printed above it in the same colour
  outlined box       one of the six gold values -- green if an entity span
                     covers the whole value, amber if it caught only a fragment
                     of it, red if nothing touched it

So the filled colours are what NER produced and the outlines are what we
wanted. Writes NER/doc*.ner.pdf.
"""

import re
import sys
from pathlib import Path

import fitz
import spacy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "StudentRecord"))
from build_gold import RECORDS  # noqa: E402

SRC = ROOT / "StudentRecord" / "Output"
OUT = Path(__file__).resolve().parent
FIELDS = ["lastname", "firstname", "street", "city", "grade", "evaluation"]

# NER label -> RGB
COLORS = {
    "PERSON":   (0.20, 0.45, 0.85),
    "GPE":      (0.15, 0.60, 0.35),
    "LOC":      (0.15, 0.60, 0.35),
    "FAC":      (0.10, 0.65, 0.65),
    "ORG":      (0.55, 0.30, 0.75),
    "DATE":     (0.90, 0.55, 0.10),
    "TIME":     (0.90, 0.55, 0.10),
    "CARDINAL": (0.85, 0.25, 0.25),
    "ORDINAL":  (0.85, 0.25, 0.25),
}
DEFAULT = (0.45, 0.45, 0.45)

HIT = (0.10, 0.55, 0.25)      # an entity span covers the whole value
PARTIAL = (0.95, 0.60, 0.00)  # an entity covers only a fragment of it
MISS = (0.85, 0.10, 0.10)     # no entity touches it at all


def find(page, needle):
    """Locate a string on the page, tolerating the line breaks NER drags in."""
    needle = re.sub(r"\s+", " ", needle).strip()
    if not needle:
        return []
    rects = page.search_for(needle)
    if rects:
        return rects
    # entity spanned a line break or a run of table padding: try each line
    for part in (p.strip() for p in needle.split(" ") if len(p.strip()) > 2):
        rects = page.search_for(part)
        if rects:
            return rects
    # single word broken by TeX hyphenation ("Satis-factory"): match a prefix
    if " " not in needle and len(needle) > 6:
        rects = page.search_for(needle[: len(needle) // 2])
        if rects:
            return rects
    return []


def label_above(page, rect, text, color, size=4.5):
    page.insert_text(
        fitz.Point(rect.x0, max(rect.y0 - 1.5, size)),
        text,
        fontsize=size,
        color=color,
        fontname="helv",
    )


def legend(page, used):
    """Colour key in the bottom margin."""
    x, y = 40, page.rect.height - 46
    page.draw_line(fitz.Point(x, y - 10), fitz.Point(page.rect.width - 40, y - 10),
                   color=(0.75, 0.75, 0.75), width=0.5)
    page.insert_text(fitz.Point(x, y), "NER: spaCy en_core_web_sm", fontsize=6,
                     color=(0.3, 0.3, 0.3), fontname="hebo")
    cx = x
    for lab in sorted(used):
        col = COLORS.get(lab, DEFAULT)
        page.draw_rect(fitz.Rect(cx, y + 4, cx + 6, y + 10), color=None, fill=col)
        page.insert_text(fitz.Point(cx + 8, y + 9.5), lab, fontsize=5.5, color=col,
                         fontname="helv")
        cx += 14 + 3.4 * len(lab)
    for dx, col, txt in ((0, HIT, "gold value: whole span found"),
                         (130, PARTIAL, "only a fragment found"),
                         (232, MISS, "not found at all")):
        page.draw_rect(fitz.Rect(x + dx, y + 14, x + dx + 6, y + 20), color=col, width=0.9)
        page.insert_text(fitz.Point(x + dx + 8, y + 19.5), txt, fontsize=5.5,
                         color=col, fontname="helv")


def annotate(pdf_path, nlp):
    doc_id = pdf_path.stem[:5]
    gold = dict(zip(FIELDS, RECORDS[doc_id]))
    doc = fitz.open(pdf_path)
    page = doc[0]

    ents = [(e.text, e.label_) for e in nlp(page.get_text()).ents]
    used_labels = set()

    # layer 1 -- what NER returned
    for text, lab in ents:
        rects = find(page, text)
        if not rects:
            continue
        used_labels.add(lab)
        col = COLORS.get(lab, DEFAULT)
        for r in rects:
            a = page.add_highlight_annot(r)
            a.set_colors(stroke=col)
            a.set_info(content=f"{lab}: {text}")
            a.set_opacity(0.30)
            a.update()
        label_above(page, rects[0], lab, col)

    # layer 2 -- what we actually wanted.
    # Coverage is judged on the entity *text*, not on rectangle overlap: an
    # entity has to contain the whole value to count as a hit, so returning
    # "315" for "315 Colborne Street" reads as partial rather than green.
    ent_norm = [re.sub(r"\s+", " ", t).strip().lower() for t, _ in ents]
    tally = {"hit": 0, "partial": 0, "miss": 0}
    for field in FIELDS:
        value = str(gold[field])
        v = value.lower()
        if any(v in e for e in ent_norm):
            state, col = "hit", HIT
        elif any(e in v for e in ent_norm if len(e) > 1):
            state, col = "partial", PARTIAL
        else:
            state, col = "miss", MISS
        tally[state] += 1

        rects = page.search_for(value) or find(page, value)
        if rects:
            for r in rects:
                page.draw_rect(r + (-1, -1, 1, 1), color=col, width=0.8)
            suffix = {"hit": "", "partial": " PARTIAL", "miss": " MISSED"}[state]
            label_above(page, rects[0], f"{field}{suffix}", col, size=4.5)
        else:
            print(f"  ! {doc_id} {field}={value!r} not locatable on the page")

    legend(page, used_labels)
    out = OUT / f"{pdf_path.stem}.ner.pdf"
    doc.save(out)
    doc.close()
    return out, len(ents), tally


def main():
    nlp = spacy.load("en_core_web_sm")
    header = f"{'document':<40}{'ents':>6}{'hit':>6}{'partial':>9}{'miss':>6}"
    print(header)
    print("-" * len(header))
    totals = {"hit": 0, "partial": 0, "miss": 0}
    for pdf in sorted(SRC.glob("doc*.pdf")):
        out, n_ents, tally = annotate(pdf, nlp)
        for k in totals:
            totals[k] += tally[k]
        print(f"{out.name:<40}{n_ents:>6}{tally['hit']:>6}"
              f"{tally['partial']:>9}{tally['miss']:>6}")
    print("-" * len(header))
    print(f"{'60 values':<40}{'':>6}{totals['hit']:>6}"
          f"{totals['partial']:>9}{totals['miss']:>6}")


if __name__ == "__main__":
    main()
