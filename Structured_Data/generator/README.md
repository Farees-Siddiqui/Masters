# Synthetic benchmark generator

Generates structured-document benchmarks from parameters rather than from a
corpus: a schema is invented to order, records are synthesised against it, and
documents are rendered from those records — so the ground truth is known by
construction instead of being annotated after the fact.

    parameters -> ER schema -> records -> documents -> ground truth
                  ^^^^^^^^^
                  Stages 1 & 2, implemented here

## Status

| Stage | What it does | State |
| :---- | :----------- | :---- |
| 1 | Read the generation parameters (domain, width, depth, seed) | done |
| 2 | Ask local Llama 3 for an ER graph and enforce those parameters | done |
| 3 | Populate the schema with records, parent-before-child | done |
| 4 | Link them into a relational graph, with nulls and orphans | done |
| 5 | Write LaTeX per document with the model, compile it, repair it | done |
| 6+ | Extraction scoring against the manifest | not started |

## Usage

```bash
cd Structured_Data/generator
python -m src.generator.cli generate --domain small_business --num-entities 4 \
    --records-per-entity 3
```

Writes `schema.json`, `instances.json`, one PDF per document and
`benchmark_manifest.json` into `out_benchmark/`. Flags:

| Flag | Default | Meaning |
| :--- | :------ | :------ |
| `--domain` | `small_business` | Domain vertical, e.g. `medical`, `education` |
| `--num-entities` | `5` | Tables/classes to request |
| `--max-depth` | `2` | Maximum nesting depth in levels (see below) |
| `--seed` | unset | Decoder seed, RNG seed; also forces greedy decoding |
| `--records-per-entity` | `5` | Data instances to generate per entity type |
| `--null-probability` | `0.05` | Chance an optional non-key field is null |
| `--orphan-rate` | `0.0` | Fraction of child foreign keys left dangling |
| `--layout-style` | `auto` | `auto`, `table`, `form` or `letter` |
| `--keep-tex` | off | Retain the `.tex` beside each PDF |
| `--max-retries` | `2` | Compilation repair attempts per document |
| `--schema-only` | off | Stop after Stages 1 & 2 |
| `--no-render` | off | Stop after Stages 3 & 4 |
| `--output-dir` | `out_benchmark/` | Created if absent |
| `--model` | `llama3.3:70b` | Model tag on the local server |
| `--base-url` | `http://127.0.0.1:11434` | Local inference endpoint |
| `--backend` | `ollama` | `ollama` (`/api/chat`) or `openai` (`/v1/chat/completions`) |
| `--max-attempts` | `3` | Retries before giving up on unusable output |

Exit codes: `2` bad arguments, `3` endpoint unreachable, `4` no usable schema,
`5` no usable records, `6` no document could be rendered. A failure in Stages
3-4 leaves `schema.json` in place, so a rerun does not start over.

**stdout carries only the paths written**, one per line: `schema.json`,
`instances.json`, `benchmark_manifest.json`, then each PDF. The summaries, the
parameters and every repair go to stderr.

Stage 5 needs `pdflatex` (or `xelatex`/`lualatex`) on PATH. Without one the
pipeline still completes: `.tex` sources and the manifest are written, nothing
is compiled, and the manifest marks every document `not_compiled` so a later run
on a machine with TeX can finish the job.

## Depth is counted in levels

A parentless entity is level 1 and a child of it is level 2, so `--max-depth 2`
permits `Invoice.customer_id -> Customer.id` and forbids a third generation
below it. `--max-depth 1` means a flat schema with no foreign keys at all.

## Enforcement, not relaying

Every later stage reads `schema.json` as ground truth, so a schema that violates
its own parameters would poison every document generated from it. What comes
back from the model is therefore repaired rather than relayed, and each repair is
recorded in `warnings` in the output and logged to stderr:

- a foreign key onto an entity that was never declared is **dropped**;
- a self-reference, a duplicate edge, a second parent for an already-parented
  child, and any cycle are **dropped** — the hierarchy has to be a forest for a
  document to be rendered from it;
- an edge that would push the hierarchy past `--max-depth` is **dropped**, and
  its child becomes another root;
- over-count is **trimmed**, preferring linked groups of entities over
  standalone ones (see `_trim_priority`; the naive roots-first policy measured
  badly against a real 70b response, keeping four parentless entities and
  discarding the only two that carried the hierarchy);
- under-count is **reported only** — the graph is still consistent, and
  re-prompting for one more entity would invalidate everything already agreed;
- a missing primary key is **synthesised**, a foreign key column named on the
  left of `->` but absent from the child's attributes is **added**, and a parent
  attribute that does not exist falls back to the parent's primary key;
- an attribute type outside the vocabulary **degrades to `string`** rather than
  failing the run.

## Stages 3 & 4: who decides what

The model supplies **field values** — the part that has to read as real domain
content. This module supplies **structure**: identifiers, which child points at
which parent, which optional field is null, which foreign key dangles.

That split is the reason `instances.json` is usable as ground truth. A join the
model invented could not be verified, and `--null-probability` /
`--orphan-rate` would mean nothing if the model were free to ignore them. All
structural choices run off one seeded RNG, so `--seed` reproduces a graph down
to which row was nulled.

Consequences worth knowing:

- Entities are populated in topological order, so a child's foreign key is
  always bound to a parent row that already exists.
- **Every `id`-typed value is assigned here, never by the model.** Not just keys
  and joins: a schema can carry a key column with no relationship behind it
  (Stage 2's forest rule drops a child's second parent but keeps the column),
  and a live run showed the cost — asked for `Enrollment.course_id`, the model
  returned `"EDU-101"`, which reads as a join to `Course` and resolves to
  nothing. Such columns are filled locally and reported.
- Foreign keys are round-robin over a shuffled parent list, not an independent
  draw per row: with 3 parents and 3 children, independent draws leave a parent
  childless about 60% of the time, and a 1:m link whose `m` is empty exercises
  nothing downstream.
- Joins live in `foreign_keys`; everything else lives in `attributes`. Nothing is
  duplicated across the two, so "is this value a join or a fact" is never a
  guess. `Record.fields()` merges them for renderers.
- Each orphaned column is named in that record's own `orphaned_keys`, so
  injected noise stays distinguishable from an extraction error. Anything
  dangling that is *not* listed there is a bug, and the tests assert it.
- A short response from the model is padded locally and a long one truncated,
  both loudly: every later stage is indexed by record count, so a short entity
  would silently shrink the benchmark.
- Keys and required fields are never nulled — a null primary key is not noise,
  it is a broken record. That also bounds what `--null-probability` can do: it
  only applies to *optional* non-key fields, and Stage 2's model tends to mark
  nearly everything `required` (one live `small_business` schema came back with
  a single optional field in seventeen). If you see fewer nulls than the rate
  implies, count the optional fields in `schema.json` before suspecting the
  rate.

## Stage 5: one document per scope

A **scope** is a root record plus every record reachable from it through
resolving foreign keys — a Customer and its three Orders become one document; a
Product with nothing under it becomes one document of its own. A child whose key
was orphaned resolves to no parent, so it becomes a root and gets its own
document: an order whose customer cannot be identified is exactly the noisy case
the corpus wants. The partition is *asserted*, not assumed — a record belonging
to no document would be a fact the manifest claims is somewhere and is not.

`benchmark_manifest.json` states, per PDF, the record ids, the raw attribute
values (not the escaped LaTeX for them), and the foreign-key join tuples
`[child, column, parent]` on that page. Documents that failed stay in the list
with `status: "failed"`, so the corpus is auditable rather than quietly short;
filter on `status == "compiled"` to get the usable set.

## The LaTeX is written by the model, not by a template

Stage 5 used to render three fixed Jinja templates. It now asks Llama 3 for the
complete source of each document (`latex_generator.py`). That buys unbounded
layout variety — the point of the corpus — and gives up a guarantee, so it is
worth being precise about which:

|  | Template | Model |
| :--- | :--- | :--- |
| Layout variety | three shapes, forever | one per document |
| Compiles | always | usually; repaired when not |
| Values verbatim | by construction | **checked, not guaranteed** |

A template *could not* alter a value: it received escaped strings and
interpolated them. A model can reword "Green Earth Landscaping", round
"$1,240.50", drop a field or invent one — and every one of those silently makes
the manifest wrong, which is worse than a failure because it looks like success.
So there are two loops, not one.

### Loop 1: it does not compile

The `.log` excerpt and the source go back to the model, up to `--max-retries`
times (default 2, so at most three compilations per document). The excerpt is the
error lines plus the few lines around each, not the whole log — the rest is font
paths, and a shorter prompt is a more accurate one.

A repair that comes back byte-identical ends the loop early: recompiling the same
input can only fail the same way.

### Loop 2: it compiles, but the data is wrong

Nothing in a compiler catches this, so the generated source is read back against
the records *before* it is compiled, and anything absent is sent back once for
restoration. Comparison is done on a de-escaped, break-flattened,
whitespace-collapsed copy of the source, so a value the model bolded, broke
across a line, tied with `~` or escaped differently still counts as present. What
it catches is a value reworded, rounded, truncated or dropped.

Restoration is attempted **once**, not looped: a model that dropped a value twice
will drop it a third time, and an unbounded loop would trade a recorded,
auditable gap for an unbounded bill. What is still missing lands in the
manifest as `fidelity.missing` on that document, and in `summary.values_missing`
and `summary.documents_value_complete` for the corpus. A page with a recorded gap
is still kept — it is a usable document with a known hole, and pretending
otherwise is what makes a corpus untrustworthy.

**`documents_value_complete` is the number that says whether the corpus is
usable.** `compiled` only says the PDFs exist.

### Loop 3 has no loop: a value that was invented

The two loops above catch a page that fails to build and a page that is missing
something. Neither catches a page with something *extra* on it — and that is the
worst case of the three, because the manifest cannot be wrong about an invented
fact, it simply never mentions it. Nothing downstream can tell it from data.

This is not hypothetical. On an `education` run at `--seed 7`, `llama3.1:8b`
wrote a student enrolment letter stating *"the account balance on file is
\$1,240.50"*. No record held that value. It came from the `letter` layout's own
example sentence, which used to hand the model a complete, ready-to-paste
sentence with a fabricated amount in it.

Two changes, both needed:

* The prompts no longer contain a usable sentence. The `letter` instruction now
  *describes* what a sentence has to do — name a field in ordinary words and
  give its exact value — instead of showing one. The escaping rules still need
  concrete characters to demonstrate, so an explicit rule carries the rest: the
  examples show form only, and none of their values may appear on the page. The
  schema prompt has had a de-priming test since Stage 1; the LaTeX prompt now
  has one too.

  The first attempt at this was worse and is worth recording. Replacing the
  worked example with a *template* — `"The <field label> on file is <the exact
  value>."` — traded one failure for two: `llama3.1:8b` wrote the angle brackets
  onto the page HTML-escaped (`\&lt;2021-02-15\&gt;`) and filled the slots
  mechanically with the wrong values, so two fields went missing. A fill-in-the-
  blank sentence is an instruction to copy. There is a test asserting no layout
  instruction contains a `<` placeholder.
* `leaked_examples()` reads the page back for the prompt's own example literals
  and reports any that no record holds, as `fidelity.leaked` on the document and
  `summary.examples_leaked` for the corpus.

There is no repair pass, deliberately. A leak is a sentence the model composed,
not a substitution, so there is nothing mechanical to put back — and a second
generation would be a fresh roll of the dice rather than a fix. It is reported
and the page is kept, because a usable document with a recorded extra fact beats
a corpus that hides one.

The detector only knows the literals *these* prompts contain, so it is a
regression guard, not a general hallucination check. A model that invents a
plausible amount of its own will not be caught by it — which is a limit of the
approach and the reason the prompt fix matters more than the detector.

### What the model is told, and why

The prompt rules are almost all observed failures, not precautions. Measured on
`llama3.1:8b`, `education`, four documents:

| | Compiled | Repairs |
| :--- | :--- | :--- |
| Before the rules below | 2 of 4 | 3 |
| After | **4 of 4** | 0 |

And on `--domain education --seed 7`, six documents, after the leak fix and the
`\\` rule below it: **6 of 6 compiled, 0 repairs, 0 leaks.** Two record
identifiers were still not printed on one letter and are recorded as
`fidelity.missing` — printing an id in running prose is the weakest spot left.

Seeded decoding is only deterministic for a *fixed* prompt, so every edit here
reshuffles which documents succeed. These numbers say the named failure stopped
happening; they are not a controlled comparison.

- **Only these packages exist**: geometry, array, booktabs, longtable, tabularx,
  parskip, enumitem, fontenc, inputenc. The repair loop cannot install one.
- **No letter-class commands.** `\begin{letter}`, `\opening`, `\closing` need
  `\documentclass{letter}`; in an article they give *Environment letter
  undefined*. The 8b reached for them for the letter layout and two repairs did
  not talk it out of them.
- **Every table row ends with `\\`, rules go on their own line.** A row missing
  its `\\` before `\bottomrule` gives *Misplaced \noalign*.
- **`&` only separates columns inside a tabular.** Outside one it must be written
  `\&` and puts a literal "&" on the page — the 8b used `\&` to separate a label
  from its value, producing `Title & Early Childhood Education` as running text.
  That compiles and passes fidelity, and is still a character that was never in
  the data.
- **Escaping**, spelled out per character, since the model now does it itself.
- **Fidelity**: reproduce values character for character, invent nothing, print
  a blank field as an em dash.

Two things are still enforced in Python rather than asked for, because asking
does not make them certain:

- **The hyphenation guard** (`\hyphenpenalty=10000`) is inserted into every
  preamble that lacks it. A hyphen LaTeX adds to a value — `Othertown` set as
  `Other-town` — is a character that was never in the data, and the value can no
  longer be read back off the page. Measured on a real 70b run.
- **`-no-shell-escape`**, and the source is extracted from between
  `\documentclass` and the last `\end{document}`, so commentary around the
  document is dropped rather than fed to the compiler.

### Ephemeral compilation

Every attempt in the repair loop happens inside one
`tempfile.TemporaryDirectory`, so the `.aux`/`.log` litter of every pass is
purged by the context manager rather than swept up afterwards. Only the `.pdf` is
moved out — plus the `.tex` when `--keep-tex` asks. A failed document keeps its
`.tex` and `.log` regardless of the flag, since the manifest's log excerpt is
unreadable without them. `pdflatex` runs under `-interaction=nonstopmode` with a
timeout; a second pass runs only when LaTeX asks for one. A PDF that appears
*despite* errors in the log is reported as such.

## Attribute types

Semantic rather than SQL-shaped, because Stage 4 has to *render* each value:
`currency` tells it to write `$1,240.50` where `decimal` would only say
`1240.5`.

    string  text  integer  decimal  currency  percent  boolean  date
    datetime  email  phone  url  address  id  enum

Common aliases (`varchar`, `money`, `uuid`, `decimal(10,2)`, …) are mapped;
anything else becomes `string` with a warning.

## Layout

    generator/
      src/generator/
        cli.py                 typer app; one command per stage
        schema_generator.py    ParametricSchemaGenerator + parameter enforcement
        schema_types.py        Attribute / EntitySchema / Relationship / SchemaGraph
        instance_generator.py  ParametricInstanceGenerator + topological_order
        instance_types.py      Record / InstanceGraph
        latex_generator.py     LLMLaTeXGenerator / escape_latex / fidelity / leaks
        renderer.py            LaTeXRenderer / compile-repair loop / manifest
        llm_bridge.py          loads layout_pipeline's LocalLLMClient, adds --seed
      tests/test_generator_schema.py     stages 1 & 2
      tests/test_generator_instances.py  stages 3 & 4
      tests/test_generator_renderer.py   stage 5

`llm_bridge` reuses the extraction side's client rather than keeping a second
copy: same server, same wire protocols, same forgiving JSON extraction. It is
loaded from its file path because `layout_pipeline` is not an installed package
and its inner package is also called `src`, which would shadow this one.

## Tests

```bash
cd Structured_Data/generator
python3 -m unittest discover -s tests -t .
```

248 tests. Every LLM call is mocked, so no server, GPU or weights are needed.

Stage 5's compilation is tested twice over: through a scripted stand-in for
`pdflatex`, so the repair loop can be driven to any outcome without a TeX
installation, and against the real binary where one is present. The tests that
need TeX skip with a clear reason when it is absent.
