# Structured Data — open-schema key-value extraction

Turn documents (papers, scans, emails, articles) into structured **records** —
`{key, value}` pairs that capture the essence of the document. The key is
usually *inferred* (nobody writes "architecture:" in the text); the value is
usually a span that appears in the text.

## Stage 1 (this): LLM extractor + see the mess

`extract.py` runs Claude structured-output extraction over a document and emits
grounded records. Every record carries a verbatim `evidence` quote and a
`grounded` flag (checked by string match against the source), so nothing is
un-traceable. Run it on a handful of docs, look at the keys that emerge — that
tells us what the next stages (key canonicalization, metadata rules) need to be.

## Setup

```bash
pip install -r requirements.txt
```

Pick a backend (Groq is free and the default):

| Backend | Env var | Cost | Get a key |
|---|---|---|---|
| `groq` (default) | `GROQ_API_KEY` | free (rate-limited) | https://console.groq.com |
| `ollama` | — (local server) | free / private | https://ollama.com |
| `anthropic` | `ANTHROPIC_API_KEY` | paid | https://platform.claude.com |

```powershell
$env:GROQ_API_KEY = "gsk_..."
```

## Run

```bash
python extract.py samples/resnet.txt                       # groq (free)
python extract.py some_paper.pdf --backend ollama
python extract.py some_paper.pdf --backend anthropic --out records.json
```

## Layout

- `schema.py` — the `Record` / `Extraction` pydantic models
- `llm.py`    — backend-agnostic structured extraction (groq / ollama / anthropic)
- `extract.py`— read (txt/pdf) → chunk → extract → verify grounding → dedupe
- `samples/`  — example inputs

## Roadmap

1. **(done)** grounded LLM extractor — reveal the emergent schema
2. key canonicalization — embed emergent keys, cluster into a controlled vocab
3. value normalization / dedup across a corpus
4. tiny gold set + precision/recall eval harness
5. metadata extractor (rules/GROBID) for bibliographic fields
