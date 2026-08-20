# layout_pipeline — PaddleOCR + XY-Cut++

Document text extraction with reading-order recovery. PaddleOCR does
detection/recognition; XY-Cut++ (arXiv:2504.10258) recovers the order in which
a human would read the blocks.

```
layout_pipeline/
  main.py               CLI
  src/ocr_engine.py     PP-OCR det+rec  +  PP-DocLayout labels
  src/xycut_plus.py     the algorithm (pure geometry, no paddle import)
  src/pipeline.py       orchestration -> ordered blocks -> JSON / Markdown
  tests/                40 unit tests, no GPU required
```

## Running

```bash
# PaddleOCR only lives in env_paddle
ocr_venvs/env_paddle/bin/python layout_pipeline/main.py \
    --input_path arxiv_papers/bert.pdf --output_dir out/

# tests need no paddle and no GPU
python3 -m unittest discover -s layout_pipeline/tests -t layout_pipeline
```

`--input_path` takes a PDF, an image, or a directory of either. Key knobs:
`--beta` (Eq. 1 cross-layout width scale), `--min_gap` (minimum whitespace in px
that counts as a split), `--theta_v` (Eq. 5 density threshold), `--dpi`,
`--device`, `--batch_size`, `--format json,md`.

## Does it work?

Page 1 of the ten papers in `arxiv_papers/`, scored as CER against the
`pdfplumber` ground truth in this repo's engine survey. Raw det+rec is the same
OCR text in raw detection order — so the delta isolates reading order alone.

| paper | cols | raw det+rec | + XY-Cut++ | delta |
|---|---|---|---|---|
| bert | 2 | 0.708 | **0.005** | −0.703 |
| resnet | 2 | 0.674 | **0.059** | −0.614 |
| faster-rcnn | 2 | 0.494 | **0.035** | −0.459 |
| densenet | 2 | 0.487 | **0.100** | −0.387 |
| unet | 1 | 0.002 | 0.002 | 0.000 |
| vgg | 1 | 0.020 | 0.044 | +0.023 |
| vit | 1 | 0.022 | 0.048 | +0.026 |
| adam | 1 | 0.004 | 0.038 | +0.035 |
| attention | 1 | 0.008 | 0.097 | +0.089 |
| batchnorm | 2 | 0.387 | 0.612 | +0.225 |
| **mean** | | **0.281** | **0.104** | **−0.177** |
| mean excl. batchnorm | | 0.269 | **0.048** | −0.221 |

Two-column pages are transformed; single-column pages, where raw detection order
was already near-perfect, give up a little.

**batchnorm is not a fair row.** Its *ground truth* is scrambled: the
column-aware reconstruction in `../prepare_pages.py` places the right-hand
author ("Christian Szegedy") after the left column's abstract, and welds one
line out of both columns (`shift, and address the problem by normalizing layer
in- requires careful tunin`). The pipeline's output for that page may well be
better than the reference it is scored against.

## Three deviations from the paper, all measured

The paper's settings are available via `--paper_defaults`. They are not the
defaults, because on this corpus each one measurably hurts.

**1. Cross-layout masking (Eq. 1–2) is off.** On the BERT page, enabling it
masks the centred author line, affiliation and email along with the title, and
Phase 4 re-anchors all four *after* the entire left column — the document title
lands at reading position 9 of 13. Off, the same page reads title → authors →
affiliation → email → Abstract. Eq. 1–2 targets newspaper banners that genuinely
sit between columns; a centred academic title is already peeled correctly by the
recursive cut as the top horizontal band. `--cross_mask` re-enables it.

**2. Titles are not pre-masked.** Masking them costs accuracy on 8 of 10 pages
(mean CER 0.152 masked vs 0.115 unmasked) and helped none. `--mask_titles`
restores the paper's behaviour.

**3. Axis choice stays density-driven (Eq. 5).** Classic XY-Cut picks whichever
axis has the wider gap, which fixes side-by-side rows on single-column pages —
the Transformer author grid is otherwise read down the columns (Vaswani → Jones
→ Shazeer). But it is worth only 0.006 mean CER and it breaks a more common
case: Phase 1 masking leaves a **phantom gap** where a figure actually sits, and
that phantom outvotes the real gutter, splitting a two-column body into rows.
`--widest_gap_axis` enables it.

The first two independently reproduce conclusions already reached in
`../../AST/layout/reading_order.py`, which this module is ported from.

## Dual Extraction Engine (step 1)

`src/dual_extractor.py` sits after reading-order recovery and decides *how* each
block gets parsed. `DualExtractionRouter.route()` takes the ordered blocks and
returns one `SemanticBlock` per input block, in the same sequence.

```python
from src.dual_extractor import DualExtractionRouter
blocks = DualExtractionRouter().route(page.blocks, page_number=0)
```

`BlockType` is `TITLE | TEXT | TABLE | FORMULA | VISION | UNKNOWN`. Text is
extracted for real; `TABLE`, `FORMULA` and `VISION` are step-2 stubs that emit a
`<!-- TYPE: pending ... -->` marker, keep the OCR text as
`metadata["ocr_fallback_text"]`, and set `metadata["status"] = "pending"` so
nothing downstream mistakes a stub for a parsed table.

There is no `_handle_title`: a heading is extracted as text, and the `TITLE`
type carries the rendering distinction. Unknown, blank and `None` labels fall
back to `_handle_text` — an unrecognised block is far likelier to be prose than
a table, and routing it to text keeps its content.

The label map is deliberately *not* `xycut_plus`'s category map. That one decides
what to **mask** during reading-order recovery, so it groups `table`,
`display_formula` and `image` together as position-flexible "vision". Extraction
cares about the difference. Captions (`figure_title`, `table_title`) route to
TEXT, not TITLE, so "Figure 3: ..." never renders as a heading.

Routing all 153 blocks from the ten-paper corpus produces 118 TEXT, 27 TITLE,
5 FORMULA, 3 VISION and **zero UNKNOWN**.

### Cropping and figures (step 2)

`src/crop_engine.py` slices block regions out of the page. Every box is padded
(5px default), normalised for inverted corners, and clamped to the page, so a
detector box running off the edge yields a valid crop instead of an exception —
and a box entirely off-page fails that one block rather than the document.

With `--extract`, TABLE / FORMULA / VISION blocks are cropped to
`<output_dir>/crops/page_{page}_block_{id}.png`. VISION blocks go further: the
crop is published as `<output_dir>/figures/fig_p{page}_{id}.png`,
`parsed_content` becomes `![Figure](figures/fig_p{page}_{id}.png)`, and status is
`completed`. TABLE and FORMULA stay `pending` with `crop_path` populated for a
later structure/LaTeX model.

```bash
ocr_venvs/env_paddle/bin/python layout_pipeline/main.py \
    --input_path arxiv_papers/densenet.pdf --output_dir out/ --extract
```

Verified end-to-end. Ten-paper corpus: `VISION 3, FORMULA 5`. `densenet.pdf`
(9 pages) additionally exercises the table path on real data —
`VISION 8, TABLE 3, FORMULA 2` — with per-page filenames (`fig_p0_15`,
`fig_p5_65`, `fig_p7_105`). Crops were spot-checked visually: ResNet Figure 1
and DenseNet's full architecture table both come out tight and unclipped.

### Formula extraction (step 3)

`src/extractors/formula_extractor.py` converts a cropped formula to LaTeX.
`FormulaExtractor.extract_latex(crop)` returns a string or `None` — it never
raises, because the caller's response to a missing dependency, an absent
checkpoint, a CUDA error and an unreadable crop is identical: fall back and say
so.

The default backend is **PP-FormulaNet** (`PP-FormulaNet_plus-M`), chosen
because its weights are already cached in `~/.paddlex/official_models/` and it
runs in the same `env_paddle` as everything else — no extra venv, no download,
no second CUDA stack. A Texify/UniMERNet-style transformers checkpoint works
through the same interface (`backend_name="transformers", model_id=...`), and
any callable `image -> str` can be injected, which is how the tests run with no
weights at all.

Successful extraction wraps the result as display math:

```
$$
\Theta=\operatorname{a r g}\operatorname*{m i n}_{\Theta}\frac{1}{N}\sum_{i=1}^{N}\ell(\mathrm{x}_{i},\Theta).
$$
```

All 5 FORMULA blocks in the corpus now extract cleanly. What they replaced:

| block | OCR text before | LaTeX now |
|---|---|---|
| 36 | `N D 1 ∑l(xi,) i=1` | `\Theta=\operatorname{a r g}\operatorname*{m i n}_{\Theta}\frac{1}{N}\sum...` |
| 37 | `1 ∂(ci, θ) m ∂Θ` | `\frac{1}{m}\frac{\partial\ell(\mathrm{x}_{i},\Theta)}{\partial\Theta}.` |
| 40 | `cid:= F2(F1(u, Θ1), Θ2)` | `\ell=F_{2}(F_{1}(\mathrm{u},\Theta_{1}),\Theta_{2})` |
| 41 | `(cid:) F2(x, Θ2)` | `\ell=F_{2}(\mathrm{x},\Theta_{2}).` |
| 45 | `Θ2←Θ2 - m α ∑ i=1 m ∂F2(xi , Θ2) ∂Θ2` | `\Theta_{2}\leftarrow\Theta_{2}-\frac{\alpha}{m}\sum_{i=1}^{m}\frac{\partial F_{2}...` |

**A failed extraction is marked `fallback`, not `completed`.** The spec asked for
`completed`; reporting a formula that silently degraded to OCR noise as parsed
would defeat the point of the status field, so failures get their own state
alongside the `<!-- WARNING: LaTeX extraction failed -->` comment. Flip it in
`_handle_formula` if you want the literal behaviour.

`SemanticBlock.status` is a property over `metadata["status"]`, so it round-trips
through JSON with everything else. Statuses: `extracted` (text), `completed`
(figure or LaTeX), `fallback` (extraction failed), `pending` (no extractor yet —
tables).

### Table extraction (step 4)

`src/extractors/table_extractor.py` converts a cropped table to Markdown or
HTML. Backend is PP-Structure's `TableRecognitionPipelineV2` (table
classification → SLANet/SLANeXt structure → cell detection → OCR), local and
cached; any callable `image -> html` can be injected.

**The output format is chosen by the table, not by config.** A grid of plain
cells becomes a Markdown pipe table. A table with merged cells stays HTML —
Markdown has no `colspan`, and flattening a spanned header into a plain grid
would misstate the data. DenseNet's architecture table is exactly that case:
its header spans two columns per model.

`densenet.pdf` (9 pages), all 3 tables:

| block | result | format | body cell fill | size |
|---|---|---|---|---|
| 45 (architecture) | `completed` | html — merged cells | 52% | 16×11 |
| 54 (CIFAR results) | `completed` | markdown | 94% | 21×8 |
| 63 (ImageNet results) | `fallback` | — | 11% | — |

**The pipeline must not re-run layout detection on the crop.** The router's
PP-DocLayout stage already decided the region is a table; letting
`TableRecognitionPipelineV2` detect layout again on a tight crop makes it find
no table region and return nothing. Every borderless table in the StudentRecord
corpus — 6 of 6 — came back empty until `use_layout_detection=False` was passed,
after which all 6 extract cleanly. Bordered tables (DenseNet) were unaffected,
which is why the problem only surfaced on a second corpus.

**Block 63 is the interesting one.** Structure recognition returned a correct
4×3 grid with *one filled cell out of nine*, from a crop that is perfectly
legible — the backend found the table and failed to attach text to it. Extra
padding (0/20/60px) does not help, so it is a model limitation on that table,
not a cropping bug.

The first version of this reported that empty grid as `completed`, which is
worse than failing: the OCR fallback text gets discarded in favour of a table
that is blank. `TableExtractor.MIN_CELL_FILL` (default 0.3) now measures the
fraction of **body** cells carrying text and rejects the result below that. The
header alone is not evidence of a parsed table — structure recognition
routinely gets the column names and nothing else. `cell_fill` is recorded in
metadata either way.

## XML projection

`src/xml_projector.py` serialises the extracted blocks into a structured
document. The Markdown output is for reading; this is for machines — every
element keeps its `id`, its source `bbox` and its `status`, so a consumer can
trace any fragment back to the pixels it came from and can tell a parsed table
from one that fell back to OCR.

```bash
ocr_venvs/env_paddle/bin/python layout_pipeline/main.py \
    --input_path arxiv_papers/densenet.pdf --output_dir out_densenet/ --xml
```

`--xml` implies `--extract` and adds `<stem>.reconstructed.xml` alongside
`<stem>.extracted.md`. The root `title` defaults to the document's own detected
`doc_title` (override with `--doc_title`).

| BlockType | element |
|---|---|
| TITLE | `<heading level="1|2|3">` — level from the detector label |
| TEXT | `<paragraph>` |
| FORMULA | `<formula format="latex" latex="…">` holding `$$…$$` |
| TABLE | `<table format="markdown|html" rows="…" columns="…">` |
| VISION | `<figure src="…" alt="…">` |
| UNKNOWN | `<paragraph unmapped="true">` |

```xml
<document title="Densely Connected Convolutional Networks" pages="9" blocks="150">
  <page number="0">
    <heading level="1" id="0" bbox="449.0,290.0,1204.0,335.0" status="extracted"
             label="doc_title">Densely Connected Convolutional Networks</heading>
    ...
    <table format="html" rows="16" columns="11" merged_cells="true" id="45"
           bbox="231.0,195.0,1407.0,758.0" status="completed" label="table">...</table>
    <table format="none" id="63" bbox="188.0,237.0,567.0,472.0"
           status="fallback" label="table">...</table>
```

Escaping is ElementTree's, not hand-rolled. Raw HTML from a table is stored as
**escaped text rather than CDATA** — ElementTree cannot emit CDATA, and an
escaped string parses back byte-identical, which is what round-tripping needs.
Characters XML 1.0 cannot represent at all (most control bytes) are stripped
rather than escaped, since one of them makes the whole document unparseable.

Verified on both output sets — well-formed, every element carrying `id`/`bbox`/
`status`, ids strictly increasing so document order matches the block array:

| output | pages | blocks | elements |
|---|---|---|---|
| `out_densenet/densenet.reconstructed.xml` | 9 | 150 | 13 heading, 124 paragraph, 8 figure, 3 table, 2 formula |
| `out/pages_only.reconstructed.xml` | 1 | 153 | 27 heading, 118 paragraph, 5 figure, 5 formula |

**`out/` reports one page, and that is a real wart.** Its input is a directory
of ten unrelated single-page images, each of which is page 0 of its own file, so
they collapse into a single `<page number="0">` and the root `title` is
whichever paper sorts first. For an actual document (`densenet.pdf`) the page
grouping and title are correct. Projecting a directory of separate documents as
one XML document is not meaningful; give each its own run, or the projector
needs a per-source grouping key it does not currently have.

## Dynamic IE engine (step 1)

`src/ie_engine/` discovers each document's *own* schema instead of imposing one.

```
blocks -> render -> local Llama -> open JSON -> DynamicDocument
```

- **`node_schema.py`** — `DynamicElement(tag_name, attributes, text_content,
  children)`. Plain dataclasses, not pydantic: a model with declared fields is
  precisely what this engine must not have. Scalars become attributes, nested
  objects become children, arrays become repeated children. Tag names are
  normalised to legal XML names at construction (`"Student Record"` →
  `student-record`, `"2024 grades"` → `n2024-grades`) so the tree can be
  serialised downstream.
- **`llm_client.py`** — `LocalLLMClient` over Ollama's `/api/chat` (default,
  `llama3.3:70b`) or any OpenAI-compatible `/v1/chat/completions`, so vLLM/TGI
  is a flag. Built on stdlib `urllib`: neither `openai` nor `requests` is
  installed in any venv here and the JSON-over-HTTP contract is small enough
  that adding them buys nothing. Temperature 0 — a schema that changes between
  identical runs is not a schema.
- **`dynamic_extractor.py`** — `DynamicInformationExtractor.extract(blocks)`.
  Long documents are chunked on paragraph boundaries rather than truncated.

**The prompt's examples come from a deliberately distant domain (shipping
manifests).** The first draft illustrated JSON shape with
`{"last_name": "Siddiqui", ...}` and `"report_card"` — and *Siddiqui appears
verbatim in doc01 and doc08 of the StudentRecord corpus*. An example that close
to the target domain can prime the model into echoing it, which would look like
successful extraction. A test now asserts no corpus term reaches the prompt.

Nothing raises. An unreachable endpoint, empty output, prose instead of JSON, or
JSON of an unexpected shape all yield an empty `DynamicDocument` with the reason
in `metadata["status"]`/`["reason"]` — a document that failed extraction must be
distinguishable from one that had no content, and neither should stop a batch.

Verified live against the running Ollama server (`llama3.1:8b`) on
`doc08_report_card`, with every value checked against the source:

```json
{"tag_name": "student_progress_report",
 "attributes": {"institution_name": "MILTON COLLEGE",
                "academic_year": "2025–2026", "student_name": "Abebe, Selam"},
 "children": [{"tag_name": "grades",
               "attributes": {"grade": "85", "standing": "Satisfactory"}},
              {"tag_name": "mailing_address",
               "attributes": {"street_address": "1180 Fanshawe Park Road",
                              "city": "London"}}]}
```

Every tag is the model's own invention — none was supplied. No hallucinated
values, and the boilerplate disclaimer was correctly dropped. Notably it bound
`Abebe, Selam` to `student_name` even though reading order had separated that
label from its value by an intervening title and table.

### Two XMLs, structural and semantic

`--mode {structural,semantic,both}` (default `both`) selects which projections
run. Both are produced by a plain invocation:

```bash
ocr_venvs/env_paddle/bin/python layout_pipeline/main.py \
    StudentRecord/Output/doc01_email.pdf --output_dir out_mode/
# -> doc01_email.reconstructed.xml   layout blocks, provenance-tagged
# -> doc01_email.semantic.xml        discovered entity tree
```

The input path is accepted positionally or as `--input_path`. `--mode
structural` is the LLM-free path; `--ie_model` / `--ie_base_url` point the
semantic path at a different model or server.

`XMLProjector.project_dynamic_xml(root_element)` does the serialisation. It
imposes **no element vocabulary at all** — every tag came from the model — and
handles what a JSON-derived tree actually contains: `True` becomes `"true"`
rather than Python's `"True"`, a stray nested container is JSON-encoded rather
than `repr`'d, empty attributes are dropped, and tags and attribute keys are
re-sanitised (a tree rebuilt from JSON can carry raw keys, and an illegal tag
otherwise raises deep inside ElementTree with no clue which node caused it).

`doc01_email.pdf`, verified against the source — every value present, no
hallucinations:

```xml
<?xml version='1.0' encoding='utf-8'?>
<email from="r.delacroix@milton-college.ca" date="12 March 2026"
       subject="year-end file" title="doc01_email" source="doc01_email.pdf">
  <to>academic.advising@milton-college.ca</to>
  <body>
    <students name="Farees Siddiqui" address="14 Fake Street, Milton"
              grade="90" evaluation="Satisfactory" />
  </body>
</email>
```

The conversational scaffolding the prompt asks it to discard — "Hi Dana",
"Closing out the last of the year-end files", "it may be worth a phone call",
"Thanks", the sign-off — is all gone, and the facts those sentences surrounded
are kept. Compare with `reconstructed.xml`, which faithfully preserves every
paragraph and its bbox: the two files answer different questions, which is why
both exist.

**When the endpoint is unreachable the run degrades to structural-only** with
one message, and writes *no* `semantic.xml` rather than an empty one — a missing
file is honest, an empty entity tree looks like a document with no content.
The endpoint is probed once per run, not once per document.

## Design notes

**Why two detectors.** det+rec supplies the text (Word F1 0.978 in the engine
survey, the highest measured, at ~3 s/page) and PP-DocLayout supplies the labels
that Phases 1 and 4 need — masking "titles and visual elements" and ranking by
label priority are both impossible without them. PP-StructureV3 would give
labels in one pass but was measured dropping ~21% of words.

**Text is never silently dropped.** A line inside no detected block becomes its
own synthetic block rather than disappearing, so a layout-recall gap cannot
delete text. `synthetic: true` marks these in the JSON.

**Lines inside a block are sorted by rows, not by the recursive cut.** Within a
block the lines are one vertical stack; the cut prefers a column split wherever
a vertical gap exists, so a single short line's ragged right edge opened a
phantom column. That put the citation year "2017" ahead of its own sentence on
the ViT page. Fixing it took vgg from 0.124 to 0.044 and vit from 0.128 to 0.048.

## Known limitations

- Single-column pages with side-by-side content (author grids) are read
  column-major. `--widest_gap_axis` fixes those specifically; see deviation 3
  for why it is not the default.
- Reading order is only as good as PP-DocLayout's blocks. A merged or missed
  region cannot be repaired downstream.
- Rotated margin text (arXiv stamps) is recognised as garbage by OCR and
  re-inserted as an `aside_text` block. It is labelled, so it can be filtered.
