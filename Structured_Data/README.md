# Structured Data — open-schema key-value extraction

Turn documents (papers, scans, emails, articles) into structured **records** —
`{key, value}` pairs that capture the essence of the document. The key is
usually *inferred* (nobody writes "architecture:" in the text); the value is
usually a span that appears in the text.

**No key is ever hard-coded.** The extractor must work on document types nobody
has seen, so the model is never handed a fixed list of allowed keys. Keys come
from the model; consistency comes from clustering them after the fact.

## Stage 1 (done): LLM extractor + see the mess

`extract.py` runs structured-output extraction over a document and emits
grounded records. Run over one paper three times it produced **44 distinct keys,
only 15 of which appeared in all three runs** — the rest were the model
inventing synonyms for keys it already had (`layers`/`layer`/`number_of_layers`
vs `depth`). That drift, not extraction quality, is why record counts swung
between runs (295 / 246 / 196).

## Stage 2 (this): induced vocabulary + exact spans

**`vocab.py` — key canonicalization by induction.** Cluster the keys the model
actually emitted, label each cluster with its most frequent member. Nothing is
authored by hand, so an invoice corpus induces invoice keys.

Keys are embedded **together with their values**, and that detail is load-bearing.
Clustering key *names* alone has no working threshold: `depth` never merges with
`layers` before `author` merges with `journal`. Clustering by value distribution
alone over-merges `dataset`/`organization`/`venue`, which all hold short proper
nouns. The concatenation of both separates them. On the sample corpus that
yields 44 → 32 keys with only correct merges:

```
architecture <- model, model_component     year   <- date
venue        <- competition, event         depth  <- layers
result       <- metric                     layer  <- number_of_layers
identifier   <- reference                  method <- technique
publication  <- journal                    hyperparameter <- training_parameter
```

A key that matches no cluster is **kept as itself** — new document types grow
the vocabulary rather than being force-fit. `raw_key` is preserved on every
record, so canonicalization is always reversible.

```bash
python vocab.py records*.json --out vocab.json      # induce
python extract.py paper.pdf --vocab vocab.json      # apply (as a hint, not a constraint)
```

**Exact character spans.** The model quotes `evidence`; `extract.py` locates that
quote and derives real offsets, so grounding is `text[start:end] == span.text`
rather than a boolean from a fuzzy substring test. Matching happens in a
markup-stripped *view* of the text with a map back to true offsets, so OCR
artifacts don't defeat it: line wraps (`im-\nprovement`), markdown emphasis
(`**3.57%**`), quotes, table pipes, citation brackets (`VGG nets [41]`) and
LaTeX (`$112 \times 112$`, `\(6.0\%\)`) are all skipped while offsets stay true.

**Grounding is reported by strength, not as one boolean.** Collapsing the ways a
record can match inflates the rate, so `grounded_by` records which path hit:

| `grounded_by` | meaning | dedup run |
|---|---|---|
| `value_in_evidence` | value located inside its own evidence quote | 167 |
| `value_in_document` | value located, but not where the model said | 19 |
| `evidence_only` | **value never found** — only the sentence was | 16 |
| `none` | nothing located | 1 |

`value_grounded_count` (the first two rows, 91.6%) is the number to quote.
`grounded_count` includes `evidence_only` and reads ~99%, which is misleading.

`evidence_only` is not a hallucination bucket — it is where the model
*transformed* a value rather than copying it: expanding `{kahe, v-shren}@ms.com`
into individual addresses, or attaching a unit from a column header so a table
cell `28.54` becomes `28.54%`. Those are usually correct and often desirable.
On this document exactly one record was genuine junk (`author: "unknown"`,
evidence `"no author mentioned"`) and it landed in `none`.

**One record per fact.** Records collapse on (key, value) and every occurrence
is kept as a `Span` in `spans`, with `mentions` as the count. `record_count`
therefore counts distinct facts — a property of the document — instead of how
talkative the model felt on a given chunk.

## Stage 3: table extraction — parsing, not inference

`tables.py` reads records straight out of markdown tables. The column header
*is* the key, the cell *is* the value, the row label *is* the entity:

```
|  model        | top-1 err. | top-5 err. |     -> {entity: "ResNet-50",
|  ResNet-50    | 22.85      | 6.71       |         key: "top_1_err", value: "22.85"}
```

No model, no hard-coded keys — every key is read off the document's own header
row. Output is a function of the input: two runs are **byte-identical**.

On `samples/resnet.mistral.md`: **364 records from 15 tables, 59 keys, 49
entities, 0 span mismatches.** The LLM stage found only **52** records in those
same table regions and exactly matched spans on **29** of them, so parsing
recovers ~7x more from tables than inference did.

It is also more *correct* where they disagree. The LLM emitted
`metric: "top-1 err."` — extracting a column header as a value — and truncated
`VGG-16 [41]` to `VGG-16`. It also read cell `28.07` and emitted `28.07%`, a
string that appears nowhere in the document and therefore cannot be grounded;
the parser keeps the cell verbatim and puts the unit in the key, where the
document put it.

```bash
python tables.py samples/resnet.mistral.md --out records_tables.json
```

## Setup on a fresh machine (e.g. GPU server)

```bash
pip install -r requirements.txt
```

Extraction runs **locally**, via Ollama:

```bash
ollama serve &
ollama pull llama3.3:70b
```

Only OCR needs a key, and only for PDFs — `.txt`/`.md` inputs need nothing.
Secrets are **not** committed; the code reads the env var first, then falls back
to `MISTRAL_API_KEY.txt` if you'd rather drop a key file in this folder.

```bash
export MISTRAL_API_KEY="..."         # OCR only — https://console.mistral.ai (paid)
```

### Extraction backends

| Backend | Requires | Cost | Notes |
|---|---|---|---|
| `ollama` (default) | local server | free / unlimited | `llama3.3:70b` |
| `anthropic` | `ANTHROPIC_API_KEY` | paid | `claude-opus-4-8` |

**Why local only.** A hosted endpoint adds variance nothing downstream can
account for: served weights, quantization and batching are outside our control
and can change between runs without notice, which surfaces as unexplained swings
in record count. Local inference pins all of it, and there are no daily token
caps to truncate a run halfway. The project previously defaulted to a hosted
Groq endpoint; that was removed for exactly this reason.

## Run

```bash
python extract.py samples/resnet.txt                          # llama3.3:70b, local
python extract.py paper.pdf --parser mistral --vocab vocab.json --out records.json
python extract.py paper.pdf --no-dedupe --out records_raw.json
```

## Layout

- `schema.py` — `Record` (model output) / `Span` / `GroundedRecord` / `Extraction`
- `tables.py` — deterministic key-value records parsed out of markdown tables
- `vocab.py`  — induce a key vocabulary by clustering emitted keys with their values
- `llm.py`    — backend-agnostic structured extraction (local ollama / anthropic)
- `extract.py`— read (txt/pdf) → chunk → extract → resolve spans → canonicalize → collapse
- `samples/`  — example inputs

## Roadmap

1. **(done)** grounded LLM extractor — reveal the emergent schema
2. **(done)** key canonicalization — embed emergent keys, cluster into an induced vocab
3. **(done)** exact character spans; collapse mentions into facts
4. **(done)** table extraction — `tables.py`, 364 deterministic records
5. **explicit `Key: value` structure** — colon-delimited lines, definition lists,
   labelled fields. Also pure parsing
6. **syntactic key inference** — for a typed literal with no stated key, take the
   key from the head noun governing it (`3.57% top-5 error` -> `error`) via a
   dependency parse. The key comes from the document's own syntax, so nothing is
   hard-coded and nothing is guessed
7. value normalization across a corpus — `ImageNet` / `the ImageNet dataset`
   still survive as two facts; same induction treatment as keys

The direction is to shrink what the LLM is responsible for until it only handles
what genuinely requires inference (`residual nets` -> `architecture`, where no
syntax states the key), and to make everything else a function of the input.
