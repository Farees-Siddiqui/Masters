# Annotation decisions — `samples/resnet.pdf`

Conventions used when hand-annotating key-value pairs from the PDF. Written so
the choices are reviewable and reversible rather than buried in the output.

**Source of truth is the PDF only.** Values are read off the rendered pages.
`samples/resnet.mistral.md` (the pipeline's OCR) is not consulted, so where the
OCR is wrong the annotation still says what the paper says — `aero` not `aeo`,
`conv2_x` not `conv2.x`, `Montúfar` not `Monttífar`, `3OJ4OJ` not `30J40J`.

---

## Settled

### 1. Register

Keys are Title Case. Values are short spans, taken as printed.

```
{Dataset: ImageNet}     {Depth: 152}        {Architecture: VGG nets}
{References: [41]}      {Error: 3.57%}      {Task: Classification}
```

The unit lives in the key, not the value: the paper's "a depth of up to 152
layers" is `{Depth: 152}`, not `{Depth: 152 layers}`. A `%` that is part of the
printed number is kept (`{Error: 3.57%}`); a bare table cell stays bare
(`{Error: 22.85}`).

### 2. Deduplication — one record per distinct pair

`{Dataset: ImageNet}` is recorded **once for the whole paper**, not once per
mention. The count is therefore a count of distinct facts, not of sentences.

### 3. Citation groups — split

*You left this one open; this is my call, overrule it here.*

The paper cites in groups: `[41, 44, 13, 16]`. Each reference number becomes its
own record, because your example `{References: [41]}` is a single bracket and
because splitting keeps the value set bounded — with deduplication there are at
most 50 `References` records rather than one per distinct grouping.

```
"the leading results [41, 44, 13, 16]"
  ->  {References: [41]}  {References: [44]}  {References: [13]}  {References: [16]}
```

### 4. Output

`Resnet/sec_*.tex`, compiled to `Resnet/Output/resnet_kv.pdf`. The rendered
tables carry a **Page** column. That is annotation metadata for review, not part
of the pair — it records where the pair was first seen in the PDF.

---

## Open — flagged for you, not decided by me

### A. The flat register cannot express table rows

This is the significant one. Your examples are flat `{Key: Value}` with no
subject. Table 3 says ResNet-50 scores 22.85 top-1 and 6.71 top-5. Annotated
flat, that becomes:

```
{Error: 22.85}   {Error: 6.71}
```

The model is gone. Every number in Tables 1–14 has this problem — roughly 400 of
the paper's numbers are table cells whose meaning is the row they sit in. As
annotated, they are recorded but not attributable.

Three ways out, none of which I applied:

| option | example | cost |
|---|---|---|
| add a subject field | `{ResNet-50, Error: 22.85}` | no longer flat `{K: V}` |
| compound the key | `{ResNet-50 top-1 err.: 22.85}` | keys stop being reusable across the corpus |
| leave tables out | — | drops most of the paper's quantitative content |

Currently: annotated flat, so the numbers are present but the row association is
lost. Say which option you want and I will redo the table sections only.

### B. The bibliography is included

Page 9 contributes `{Author: ...}`, `{Title: ...}`, `{Venue: ...}`, `{Year: ...}`
for all 50 references. It is part of the PDF, so it is annotated — but it is a
large share of the total, and it is about *cited works* rather than about this
paper. It lives in `sec_references.tex` alone so it can be dropped wholesale.

### C. Spelled-out numbers are kept as printed

The introduction writes "a depth of sixteen [41] to thirty [16]", so the records
are `{Depth: sixteen}` and `{Depth: thirty}` rather than `16` and `30`. Consistent
with taking values as printed, but it means `Depth` holds both numerals and
words.
