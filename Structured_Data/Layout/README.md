# LayoutLMv3 probe

`nielsr/layoutlmv3-finetuned-funsd` over the ten student-record PDFs, drawn onto
the pages the same way as `NER/`.

```
python Layout/run_layoutlmv3.py     # -> Layout/*.lmv3.pdf
```

Word boxes come from the PDF itself (PyMuPDF), so `apply_ocr` is off and the
model receives exact text with true coordinates — no OCR error in the loop.

## What it is being asked

FUNSD's label set is not our schema. It is `HEADER / QUESTION / ANSWER / OTHER`,
so the model never says `lastname`. The question here is the prior one:

> can it tell which spans on the page are keys and which are values?

That is steps 1–2 of the reversal. Naming and grouping are still open.

## Reading the pages

| mark | meaning |
|---|---|
| blue highlight | QUESTION — the model thinks this span is a key |
| green highlight | ANSWER — a value |
| purple highlight | HEADER |
| green outline | a gold value the model tagged ANSWER |
| amber outline | partly tagged |
| red outline | not tagged ANSWER |

Consecutive same-label words on a line are merged into one span, so the printed
caption marks a run rather than a word.

## Result

```
document                                    Q    A    H    O   hit  part  miss
doc01_email.lmv3.pdf                        8   10    0   84     2     0     4
doc02_letter.lmv3.pdf                       1    8    0   87     2     0     4
doc03_vertical_table.lmv3.pdf               8    8    3   13     6     0     0
doc04_horizontal_table.lmv3.pdf            13    8    0    0     6     0     0
doc05_form.lmv3.pdf                        18    9    6   18     6     0     0
doc06_prose_plus_address_table.lmv3.pdf     4    5    3   64     3     0     3
doc07_table_plus_prose_address.lmv3.pdf     5    7    1   41     4     0     2
doc08_report_card.lmv3.pdf                  6    8    3   27     5     1     0
doc09_directory_entry.lmv3.pdf              0    0    2   30     0     0     6
doc10_grouped_header_table.lmv3.pdf         9    8    1    2     6     0     0
------------------------------------------------------------------------------
60 values                                                       40     1    19
```

**40/60 values tagged ANSWER**, against spaCy's 27 "found" — and unlike spaCy's
hits, these carry a useful label, since ANSWER means *this is a value* rather
than CARDINAL meaning *this is a number*.

The split is not uniform, and the shape of it is the finding:

| documents | values recovered |
|---|---|
| 03, 04, 05, 10 — fully tabular or form | **24/24** |
| 08, 07, 06 — mixed | 12/18 |
| 01, 02 — prose | 4/12 |
| 09 — flat directory line | **0/6** |

## Failure modes worth looking at

**`doc09`** — the model tags the page title HEADER and everything else OTHER.
Not one of the six values is marked. With no key-value geometry there is nothing
for a form model to hold on to.

**`doc01`** — it handles the email *header* block perfectly (`From:`/`To:`/
`Date:`/`Subject:` as QUESTION, their values as ANSWER) because that block is
shaped like a form. In the body it finds `Farees Siddiqui` but also tags
`Closing`, `The`, `still` and `They` as QUESTION — hunting for form fields in
running prose and inventing them.

**`doc03`** — the document spaCy failed hardest on (2 entities, 5 of 6 missed)
is recovered completely: six labels blue, six values green, headers purple.

## Recovering XML

```
python Layout/recover.py -i <document.pdf> -o <output.xml|directory/>
python Layout/check_recovered.py            # how much survived the round trip
```

LayoutLMv3 tags words, consecutive same-label words merge into chunks, and each
ANSWER is linked to the nearest QUESTION — to its left on the same row, or
directly above — by greedy nearest-first assignment. Each pair becomes an
element. Output lands in `recovered/`.

Nothing is invented:

* element names are slugified from the label printed on the page, so a document
  saying "Surname" yields `<surname>`, never `<lastname>`
* a value with no linkable key becomes `<unkeyed>` rather than a guessed name
* the root is `<document>`; nothing on the page says "student-record"
* when the model finds nothing the file is an empty `<document/>`

### Result

```
document                           elems  present  keyed
doc01_email                            6      0/6    0/6
doc02_letter                           4      0/6    0/6
doc03_vertical_table                   6      6/6    6/6
doc04_horizontal_table                 6      6/6    6/6
doc05_form                             6      6/6    6/6
doc06_prose_plus_address_table         3      3/6    2/6
doc07_table_plus_prose_address         6      4/6    4/6
doc08_report_card                      5      3/6    2/6
doc09_directory_entry                  0      0/6    0/6
doc10_grouped_header_table             6      6/6    6/6
--------------------------------------------------------
60 values                                    34/60   32/60
```

`present` = the gold value appears as some element's text. `keyed` = that
element also got a name from the page. Whether the name is the *right* one is
deliberately not scored: the document says `Surname` where gold says
`lastname`, and deciding those are the same is the unsolved part.

### What the outputs show

**Four documents round-trip exactly** — 03, 04, 05, 10, all 6/6 keyed. On the
fully tabular and form surfaces this pipeline is already a working extractor.

**`doc09` produces `<document/>`** — an empty element, because the model tagged
nothing. That is the honest answer for a flat directory line, not a failure of
the output format.

**The prose documents produce the wrong record.** `doc01` recovers the email
*envelope* — `<from>`, `<to>`, `<date>`, `<subject>` — which are genuine
key-value pairs on the page, just not the student record. It also emits
`<still label="still">Farees Siddiqui.</still>`, where a stray word tagged
QUESTION became an element name. `doc02` recovers `<dear>Ms. Nguyen,</dear>`
and three unkeyed names.

**Section headings hijack keys.** In `doc05` the street came out as
`<section_2_address label="SECTION 2 — ADDRESS">9 Rosedale Court</...>` — the
heading won the link over the nearer `Street` label.

**Nothing nests.** Every output is flat. `street` and `city` never end up under
a shared parent in any of the ten, including the two documents whose layout
displays the grouping. Recovering the tree is untouched by this model.

**The same field gets three names.** `<last_name>` in doc03, `<surname>` in
doc04 and 07, `<last>` in doc10 — correct readings of what each page prints,
and exactly the vocabulary problem `vocab.py` induction exists to solve.

## Hybrid pass: GLiNER typed with the corpus's own labels

```
python Layout/harvest_labels.py                     # -> label_vocab.json
python Layout/recover.py --gliner -i <pdf|dir> -o <xml|dir/>
```

LayoutLMv3 solves the labelled surfaces and fails on prose, where there is no
key on the page to link a value to. The second pass supplies keys there without
anyone authoring a field list:

1. `harvest_labels.py` keeps every span LayoutLMv3 tagged QUESTION across the
   corpus — the labels the documents print on themselves.
2. A caption seen in **2+ documents** is kept. Agreement across documents is the
   filter, so `still`, `Dear`, `Academic` and `Closing` drop out on their own.
   What survives is `ADDRESS, City, Evaluation, First Name, Grade, Mailing
   Address, NAME, Standing, Street, Surname` — all six fields covered, no noise.
3. Those become the zero-shot type names for GLiNER over the page text.
4. Only spans the layout pass did not already claim are added, each tagged
   `by="gliner"` with its score, so provenance stays visible.

Rejected rather than repaired: a value identical to its own key
(`Evaluation` tagged Evaluation), and a value existing only inside an email
address or URL, which is how `milton-college` was read as a city. TeX's
line-break hyphenation is undone first, so `Satis-\nfactory` is one token.

### Result

```
document                           elems  present  keyed  right key
doc01_email                           10      4/6    4/6        3/6
doc02_letter                           9      4/6    4/6        3/6
doc03_vertical_table                   8      6/6    6/6        6/6
doc04_horizontal_table                 7      6/6    6/6        6/6
doc05_form                             7      6/6    6/6        6/6
doc06_prose_plus_address_table         8      5/6    4/6        4/6
doc07_table_plus_prose_address         8      6/6    6/6        6/6
doc08_report_card                      8      5/6    4/6        4/6
doc09_directory_entry                  5      5/6    5/6        3/6
doc10_grouped_header_table             7      6/6    6/6        6/6
-------------------------------------------------------------------
60 values                                    53/60   51/60      47/60
```

against **34/60 present, 32/60 keyed** for LayoutLMv3 alone. `doc09`, which
returned an empty document before, now yields 5 of 6 values.

`right key` uses a hand-written synonym table that lives in
`check_recovered.py` and is **never given to the pipeline** — inducing that
mapping is the open problem, so scoring against it by hand is only honest for
reporting.

### What is still wrong

**Names are never split.** `Farees Siddiqui` stays one span, so `lastname` and
`firstname` both miss — 5 of the 7 remaining misses. This is value
decomposition, a different problem from key-value recognition: the page shows
one slot where the schema wants two fields.

**`evaluation` lands under `grade`.** GLiNER prefers the type `Grade` over
`Evaluation`/`Standing` for `Satisfactory`, in doc01, doc02 and doc09. That is
3 of the 4 wrong keys and the reason `right key` trails `keyed`.

**False positives remain.** `<city>Milton College</city>` in doc02 reads the
institution as a city; `<surname>Dmitri</surname>` in doc09 puts a given name
under Surname. doc01 still carries `<still label="still">` from the layout pass.

## Caveat on the environment

transformers 5.5 replaced LayoutLMv3's specialized tokenizer with a generic
backend whose `__call__` drops the `boxes` argument silently, so
`LayoutLMv3Processor` produces an encoding with no geometry. `classify()`
therefore builds the encoding by hand — word box repeated across sub-tokens,
zero boxes for CLS/SEP — which is what `LayoutLMv3TokenizerFast` used to do.
If this is ever run under transformers 4.x, the processor path can come back.
