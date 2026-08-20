# NER probe

Off-the-shelf NER (spaCy `en_core_web_sm`) run over the ten student-record
PDFs, drawn back onto the pages so the failure is visible rather than tabulated.

```
python NER/annotate.py      # reads StudentRecord/Output/*.pdf -> NER/*.ner.pdf
```

## Reading the pages

| mark | meaning |
|---|---|
| coloured highlight | an entity spaCy returned; colour = its label, printed above the span |
| green outline | a gold value an entity span covered completely |
| amber outline | an entity caught only a fragment of the value |
| red outline | nothing NER returned touched the value |

Highlight annotations carry the label and full entity text, so hovering in a
PDF viewer shows what was matched.

## Result over 60 values (6 fields x 10 documents)

| | count |
|---|---:|
| whole span found | 27 |
| fragment only | 8 |
| not found | 25 |

Per-field, "whole span found" is concentrated almost entirely in `city` and the
person names; `evaluation` is never recognised and `street` is almost always a
fragment (the house number alone).

## Caveats on those numbers

**"Found" is not "identified."** A hit means some entity span contained the
value, not that the label was useful. `90` is found as `CARDINAL` — the same
label the house number and the page number get. `Milton` is `GPE` in one
document while `Waterloo`, `Hamilton` and `Mississauga` are `PERSON`.

**Some hits are over-long spans.** Doc 02 returns `'Thuy Nguyen\n88'` as one
`PERSON`, which swallows the house number and counts as a hit for both
`firstname` and `lastname` while separating neither.

**Gold values are located by string search**, so a value that also appears
elsewhere on the page gets boxed there too — `Milton` is outlined both as the
city and inside `milton-college.ca` in doc 01.

## The page worth looking at first

`doc03_vertical_table.ner.pdf`. It is the cleanest, most explicitly labelled
document in the corpus — six rows of `Field | Entry` — and spaCy returns
**two entities on the whole page**: `94` as `CARDINAL`, and `First` as
`ORDINAL`, where `First` is part of the column *label* "First Name" rather
than any value at all. Five of the six values are red.
