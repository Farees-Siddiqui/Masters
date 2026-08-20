"""Ground truth for the student-record corpus.

The ten documents in tex/ are renderings of the ten records below. The records
are the source, the documents are derived from them, so the ground truth is
correct by construction -- nothing here is annotated back out of a document.

Every record has the identical structure:

    student-record
      name        @lastname @firstname
      address
        street
        city
      evaluation  @grade  + text

Only the values differ. Writes gold/doc01.xml ... gold/doc10.xml.
"""

from pathlib import Path
from xml.sax.saxutils import quoteattr, escape

# doc id -> (lastname, firstname, street, city, grade, evaluation)
RECORDS = {
    "doc01": ("Siddiqui",   "Farees", "14 Fake Street",                 "Milton",       90, "Satisfactory"),
    "doc02": ("Nguyen",     "Thuy",   "88 Braemar Avenue",              "Guelph",       76, "Satisfactory"),
    "doc03": ("Okonkwo",    "Chidi",  "2201 Lakeshore Road",            "Burlington",   94, "Excellent"),
    "doc04": ("Kaur",       "Simran", "47 Hillcrest Drive",             "Brampton",     82, "Satisfactory"),
    "doc05": ("Martinez",   "Elena",  "9 Rosedale Court",               "Oakville",     68, "Needs Improvement"),
    "doc06": ("Fitzgerald", "Ronan",  "315 Colborne Street",            "Kitchener",    91, "Excellent"),
    "doc07": ("Yamamoto",   "Hana",   "62 Winston Churchill Boulevard", "Mississauga",  79, "Satisfactory"),
    "doc08": ("Abebe",      "Selam",  "1180 Fanshawe Park Road",        "London",       85, "Satisfactory"),
    "doc09": ("Petrov",     "Dmitri", "23 Sunnybrook Lane",             "Waterloo",     58, "Unsatisfactory"),
    "doc10": ("Chen",       "Wei",    "704 Highland Avenue",            "Hamilton",     97, "Excellent"),
}

TEMPLATE = """<student-record>
  <name lastname={lastname} firstname={firstname}/>
  <address>
    <street>{street}</street>
    <city>{city}</city>
  </address>
  <evaluation grade={grade}>{evaluation}</evaluation>
</student-record>
"""


def to_xml(record):
    lastname, firstname, street, city, grade, evaluation = record
    return TEMPLATE.format(
        lastname=quoteattr(lastname),
        firstname=quoteattr(firstname),
        street=escape(street),
        city=escape(city),
        grade=quoteattr(str(grade)),
        evaluation=escape(evaluation),
    )


def main():
    out = Path(__file__).parent / "gold"
    out.mkdir(exist_ok=True)
    for doc_id, record in RECORDS.items():
        (out / f"{doc_id}.xml").write_text(to_xml(record), encoding="utf-8")
    print(f"wrote {len(RECORDS)} records to {out}")


if __name__ == "__main__":
    main()
