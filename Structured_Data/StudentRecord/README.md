# Student-record corpus

Ten documents, one schema. Each document is a rendering of one record; the
records are in `build_gold.py` and are the source, so the ground truth is
correct by construction — nothing was annotated back out of a document.

```
student-record
  name        @lastname  @firstname
  address
    street
    city
  evaluation  @grade  + text content
```

Six leaf values per record. Every document contains all six.

## Build

```
./build.sh          # gold/*.xml from build_gold.py, Output/*.pdf from tex/
```

## The ten formats

| # | file | surface | how the values are carried |
|---|---|---|---|
| 01 | `doc01_email` | email | all six in running prose, no layout |
| 02 | `doc02_letter` | formal letter | address as a letterhead block, rest in prose |
| 03 | `doc03_vertical_table` | table, vertical | one row per field, `Label \| Value` |
| 04 | `doc04_horizontal_table` | table, horizontal | labels across the top, values beneath |
| 05 | `doc05_form` | filled form | fields grouped under numbered sections |
| 06 | `doc06_prose_plus_address_table` | mixed | address tabular, name + evaluation in prose |
| 07 | `doc07_table_plus_prose_address` | mixed | name + evaluation tabular, address in a sentence |
| 08 | `doc08_report_card` | report card | labelled header block + small results table |
| 09 | `doc09_directory_entry` | directory line | one flat line, almost no labels |
| 10 | `doc10_grouped_header_table` | table, spanning header | grouping shown by header spans |

## Conventions

**Values differ between documents, structure does not.** Each document is a
different student. Doc 01 is the original record verbatim; the other nine vary
so that nothing can score well by memorising values.

**Labels vary for the same field.** `Last Name` / `Surname` / `Family Name` /
`Last`; `City` / `Municipality`; `Grade` / `Score` / `Mark` / `Numeric Grade`.
Surface wording is not part of the structure, so recovering the tree requires
inducing that these name the same node.

**Addresses are exactly street + city.** No province or postal code, even though
real documents carry them — extra fields outside the schema would make the
correct output ambiguous.

**Casing and word order are not preserved.** Doc 08 prints the name as
`Abebe, Selam`; doc 09 as `Petrov, Dmitri — 23 Sunnybrook Lane, Waterloo — 58
(Unsatisfactory)`. Value comparison cannot be string equality.

**Gold XML is well-formed.** The original sketch had an unclosed `<name>` and an
unquoted `grade=90`; `gold/*.xml` closes and quotes both. No structural change.

## What each document does to the structure

The schema has three groupings — `name`, `address`, `evaluation` — and the
documents disagree about how much of that grouping survives:

- **shown** — 05 (section headings), 10 (spanning headers), 02 and 08 (address
  as a visual block), 06 (a table titled *Mailing Address*)
- **gone** — 03, 04, 07, 09, and the whole of 01: six fields at one level, with
  nothing on the page tying `street` to `city` or `grade` to its remark

That split is the point of the corpus. The values are recoverable from every
document; the *tree* is only displayed by some of them.
