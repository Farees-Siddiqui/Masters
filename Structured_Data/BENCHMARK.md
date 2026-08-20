# Benchmark plan

How to measure the extractor. Two tracks share one scorer: a **synthetic**
track where documents are generated from known records, and a **real** track
where one real paper has a hand-made answer key.

---

## The core move: invert the pipeline

Extraction is `document -> records`, and you cannot score it without already
knowing the answer. Generation is `records -> document`, and the answer is what
you started with. So author the ground truth first and the document second.
Gold becomes free, exact, and unlimited — there is no annotation step, ever.

```
seed records  ──render──▶  document  ──extract──▶  predicted records
   (gold)       templates   (pdf/txt/md)   the pipeline         │
      └──────────────────── score ────────────────────────────-─┘
```

**"AI as an undergrad student"** is a claim about cost and throughput, not
capability. An undergrad hand-writes 50 test documents in a week; a generator
writes 50,000 overnight at the same fidelity. The fidelity bar is low on
purpose — the generator never has to be *right*, only *faithful to the tuple it
was handed*. Correctness is delegated to the seed data.

## What varying the surface buys

One fact — `(Ken, MLL)` — has many surfaces:

| rendering | surface |
|---|---|
| narrow table | `Inst \| Ken` / `Course \| MLL` |
| wide table | header `Prof/Fac \| Course`, row `Ken Faust \| MLL` |
| prose | "Ken teaches MLL" |

The extractor should return the same record for all three. **That invariance is
the quantity being measured.** Any variance across renderings is extractor
error, attributable to the layout axis alone, because nothing else changed.

## The factorization

```
Structured Data  ≅  Shape  ×  Values  ×  Doc Layout
                  = Schema × Instance ×  Format
                  =  100   ×   100    ×   100      = 1e6
```

| axis | held fixed | what a score delta measures |
|---|---|---|
| **Schema** | values, format | whether key/entity structure itself breaks extraction — depth of the tuple graph, joins across records |
| **Instance** | schema, format | value-type sensitivity: floats vs ints, units, long vs short, proper nouns vs numbers |
| **Format** | schema, values | table vs prose vs `Key: value` — the axis that pays for the whole thing |

The crossing is not about volume, it is about **attribution**. Hold two axes
fixed, vary the third, and the score delta is caused by that axis alone. That
turns *"the extractor got 71%"* into *"it loses 20 points when a fact moves from
a table into prose, and zero when the value is a float instead of an int"* —
which points at the component to replace.

`1e6` is illustrative: it shows the space is large and factorizes cleanly. The
real N comes from a pilot batch once throughput is measured.

### This axis measures the roadmap directly

A fact rendered as a table row has its key **stated** in the header, so
`tables.py` parses it deterministically. The same fact in prose has **no stated
key at all** and must be inferred. The table-vs-prose delta is therefore a
direct measurement of roadmap items 5 and 6 — how much document surface a
parser can own before inference is genuinely required.

---

## Settled decisions

### 1. Templated renderer

Deterministic templates per format. Gold *is* the seed, exactly — no drift, no
self-report, no verification pass.

```
seed:  {entity: Ken Faust, key: course, value: MLL}

prose.tmpl  ->  "{entity} teaches {value}."
table.tmpl  ->  | Prof/Fac  | Course |
                | Ken Faust | MLL    |
kv.tmpl     ->  Course: MLL
```

Values land in the document **verbatim**, which matters more than it first
appears. `extract.py` grounds by locating the value in the source, so on
synthetic data `value_grounded` should be reachable at ~100%. That makes
`grounded_by` a sharp secondary diagnostic: any `evidence_only` here is a
genuine defect (the model transformed a value it could have copied), whereas on
a real paper it is usually *desirable*. Same signal, opposite reading, because
the input is controlled.

**Rejected:** an LLM writing prose from the tuples. It paraphrases values
(breaking span grounding) and adds flavour facts that were never seeded — so a
*correct* extraction of an unseeded fact scores as a false positive, which
quietly invalidates precision.

**The risk taken on knowingly.** Templates are deterministic and therefore easy,
and an easy benchmark reports a flattering number. The mitigation does not
require giving up determinism: make the *template inventory itself* the
difficulty gradient. Many templates per format, including adversarial ones drawn
from the failure modes the span resolver already claims to handle — value split
across a line wrap (`im-\nprovement`), unit living in the column header,
citation bracket glued to the value (`VGG nets [41]`), markdown emphasis
(`**3.57%**`), two-column reflow, footnote markers. Those become named,
reproducible test cases instead of incidental noise.

### 2. Scoring unit: `(entity, key, value)`

Full relational match, with keys matched through induced clusters rather than
string equality.

```
gold:      (Ken Faust, course, MLL)
predicted: (Ken Faust, class_taught, MLL)
                       └── same induced cluster? ── yes -> hit
```

**The gap this leaves.** `vocab.py` induces clusters from *emitted* keys, and
cluster labels are the model's most frequent member — not the gold key name.
Gold says `course`, the model says `class_taught`, nothing connects them.

**The fix that stays honest to the project's philosophy:** pool the gold keys
into the induction. Embed predicted and gold keys together with their values,
cluster as usual, and score a key match iff both land in the same cluster. The
gold key is just another key in the pool — no hand-authored crosswalk, nothing
hard-coded, and `raw_key` keeps it reversible. Same argument the README already
makes for why keys are embedded *with* their values rather than alone.

**Report three nested numbers, not one.** value hit → also key hit → also entity
hit. A single fused score cannot distinguish "found the fact, named it
differently" from "missed the fact", and those demand different fixes.

### 3. Scale: pilot first

Generate a small balanced batch (e.g. 5 schemas x 5 instances x all formats),
run it end to end, and measure two things — seconds per document through the
local 70B, and the **variance** of each axis's effect. Axes with large, stable
effects need few samples; noisy ones need more. That yields a defensible N per
axis instead of a round number.

---

## Order of work, and why

**Build the scorer first, against the ResNet gold set.**

### The scorer decides what "correct" means

Everything else depends on that decision. The generator has to hand the scorer a
gold answer in whatever shape the scorer expects. Build the generator first and
you will write gold in a shape the scorer cannot read, and redo it. It is like
writing exam questions before deciding how to mark them.

### It is the only piece that yields a real number immediately

Already in hand:

- a real document — `samples/resnet.pdf`
- a hand-made answer key — the 868 records in `Resnet/`
- an extractor that runs today, with saved output in `records.json` and
  `records_tables.json`

The only missing piece is the comparison. Build it and you can state *"the
extractor recovers X% of the facts in a real paper."* That is a result.

Build the generator first and you have 3,000 synthetic documents and no way to
grade any of them — lots of work, no number.

### Do the uncertain part first

The generator is known work: templates are fiddly but they will work. The
scorer contains a genuine unknown — whether induced-cluster key matching
actually bridges gold keys to emitted keys. If it does not, the benchmark design
changes. Better to learn that now, on 868 records that can be read by hand, than
after generating a million documents.

### Debug the ruler against something measurable by hand

When the scorer calls a record wrong, you need to check whether it is *really*
wrong or whether the scorer is broken. On ResNet you can open the PDF and look.
On synthetic data a scorer bug and an extractor bug are indistinguishable — both
just show up as a lower number.

### The catch

Tuning a scorer on one document risks tailoring it to that document. ResNet is
heavy on tables and numbers, light on prose and dates. So keep the matching
rules general, add no special case that only makes sense for this paper, and
treat the number as a baseline to re-check once synthetic documents exist — not
a final verdict.

---

## Steps

1. **Convert the gold set to JSON.** The 868 records live in `Resnet/sec_*.tex`
   — fine for a human, useless to a program. Rows are rigidly formatted
   (`\E{entity} & \K{key} & value & \Sd{src}`), so this is mechanical parsing
   into the shape of `GroundedRecord`.
   → `benchmark/tex_to_gold.py`, `benchmark/gold_resnet.json`
2. **Write the scorer.** Match records, report the three nested numbers, and
   dump a readable list of misses to eyeball.
   → `benchmark/score.py`
3. **Run it** against the saved extractor output. Baseline, plus a stress test
   of the key-matching rule.
4. **Then build the generator**, knowing exactly what format it must emit.

## Usage

```bash
python benchmark/tex_to_gold.py Resnet --out benchmark/gold_resnet.json
python benchmark/score.py --gold benchmark/gold_resnet.json \
                          --pred records.json records_tables.json \
                          --induce --misses benchmark/misses.md
```

---

## First baseline (steps 1–3 done)

868 gold records vs the saved output of both extractor stages — 189 LLM
records and 364 table records, 553 predicted in total.

| gold subset | n | value | + key | + entity |
|---|---:|---:|---:|---:|
| **all** | 868 | 30.2% | 27.2% | 8.2% |
| table-sourced (`T.*`) | 318 | 74.5% | 68.6% | 19.8% |
| prose-sourced | 550 | 4.5% | 3.3% | 1.5% |
| atomic (value ≤ 4 words) | 508 | 51.4% | 46.3% | 13.8% |
| propositional (longer) | 360 | 0.3% | 0.3% | 0.3% |

### What the run established

**Induced key matching works, and it is worth 62 points.** Table-sourced key
recall is 6.6% when keys must match as strings and **68.6%** when gold keys are
pooled into the induction. Gold `top1_error` and parsed `top_1_err` land in one
cluster without anything being hand-mapped. This was the open risk in the whole
design; it is now measured, not assumed.

**The table/prose gap is 70 points** — 74.5% vs 4.5% value recall. That is the
format axis, measured on real data before the generator exists, and it is the
strongest evidence so far for the roadmap's direction: a parser owns the table
surface outright, and inference is barely recovering anything outside it.

**Entity linkage is carried entirely by `tables.py`.** 0 of 189 LLM records
carry an entity; 281 of 364 table records do. The triple-level score is
therefore capped by the parser's coverage, and no amount of LLM improvement
raises it until the LLM stage emits `entity` at all. That is a concrete,
unglamorous defect the benchmark surfaced immediately.

### What the run exposed about the gold set

**360 of 868 gold records are propositions, not facts.** Their values are
sentences the gold author wrote (`"Deeper neural networks are more difficult to
train"`), and no extractor reproduces those verbatim, so string equality scores
them at 0.3% regardless of how good the extractor is. They are not measuring
extraction; they are measuring paraphrase.

The scorer reports atomic and propositional separately rather than hiding the
problem in an average, so **51.4% / 46.3% / 13.8% on the 508 atomic records is
the honest headline** and the other 360 are a pending decision:

1. drop them from gold — cleanest, loses real content the paper does state;
2. keep them, scored by a different rule (entailment / embedding similarity) —
   measures something real but stops being a function of the input;
3. keep them as an unscored annotation layer — honest, but they stop being gold.

This is exactly the class of problem the "scorer first" ordering existed to
find, and it would have been invisible until after a million documents had been
generated.
