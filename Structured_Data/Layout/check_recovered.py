"""How much of each record survived the round trip?

Compares Layout/recovered/*.xml against the records they were rendered from.
Reports only what can be checked without inventing a mapping between the
document's own labels and the gold tag names:

  value present   the gold value appears as some element's text
  keyed           ...and that element got a name from a label on the page
                  (whether the name is the *right* one is not scored -- the
                  document says "Surname" where gold says "lastname", and
                  deciding those are the same is the unsolved part)
"""

import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "StudentRecord"))
from build_gold import RECORDS  # noqa: E402

FIELDS = ["lastname", "firstname", "street", "city", "grade", "evaluation"]
REC = Path(__file__).resolve().parent / "recovered"

# DIAGNOSTIC AID ONLY -- hand-written, and deliberately not used by any part of
# the pipeline, which must never be handed a synonym table. It exists so the
# report can say whether a recovered value landed under a *defensible* key
# rather than merely under some key. Inducing this mapping is the open problem.
SYNONYMS = {
    "lastname":   {"surname", "last", "last_name", "family_name", "student", "name"},
    "firstname":  {"first", "first_name", "given_name", "given_name_s"},
    "street":     {"street", "street_address", "mailing_address", "address",
                   "section_2_address"},
    "city":       {"city", "municipality"},
    "grade":      {"grade", "score", "mark", "numeric_grade"},
    "evaluation": {"evaluation", "standing", "assessment", "remark"},
}


def norm(s):
    return " ".join((s or "").split()).strip(" .,").lower()


def main():
    hdr = f"{'document':<34}{'elems':>6}{'present':>9}{'keyed':>7}{'right key':>11}"
    print(hdr)
    print("-" * len(hdr))
    tot_present = tot_keyed = tot_right = 0
    for xml in sorted(REC.glob("doc*.xml")):
        doc_id = xml.stem[:5]
        gold = dict(zip(FIELDS, RECORDS[doc_id]))
        root = ET.parse(xml).getroot()
        elems = [(e.tag, norm(e.text)) for e in root.iter() if e is not root]

        present = keyed = right = 0
        missing, wrongkey = [], []
        for f in FIELDS:
            v = norm(str(gold[f]))
            hit = [t for t, txt in elems if txt == v or v in txt.split(" | ")]
            if hit:
                present += 1
                if any(t != "unkeyed" for t in hit):
                    keyed += 1
                if any(t in SYNONYMS[f] for t in hit):
                    right += 1
                else:
                    wrongkey.append(f"{f}<-{hit[0]}")
            else:
                missing.append(f)
        tot_present += present
        tot_keyed += keyed
        tot_right += right
        note = ""
        if missing:
            note += f"   missing: {', '.join(missing)}"
        if wrongkey:
            note += f"   wrong key: {', '.join(wrongkey)}"
        print(f"{xml.stem:<34}{len(elems):>6}{present:>7}/6{keyed:>5}/6"
              f"{right:>9}/6" + note)
    print("-" * len(hdr))
    print(f"{'60 values':<34}{'':>6}{tot_present:>7}/60{tot_keyed:>5}/60"
          f"{tot_right:>9}/60")


if __name__ == "__main__":
    main()
