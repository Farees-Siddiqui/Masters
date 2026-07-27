---
name: structured-data-kv-extractor
description: "Structured_Data project — open-schema key-value extraction pipeline (Mistral OCR + Groq Llama 70B); state, files, and how to resume on a GPU server"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9a954529-22da-4299-bbd0-0344c2100538
  modified: 2026-07-27T18:10:19.620Z
---

New project `C:\Users\faree\Documents\Masters\Structured_Data` (separate from the [[project-overview]] AST project, but reuses AST's Mistral OCR pattern). Goal: parse documents (PDF/scans + unstructured text) into **records** = `{key, value}` pairs capturing a doc's essence. Keys are usually **inferred** (e.g. "residual nets" → `architecture`, a word absent from the text); values are usually spans in the text. Mental model (from user's notes): a record = one row of a relational table; records join via foreign keys → database → analysis. **Currently only building stage 1, the key-value extractor.**

**Pipeline (built, working):** parse → chunk → LLM extract per chunk → pydantic validate → dedupe → grounding-check.
- `parsers.py` — `parse(path, parser)`; `pypdf` (naive, garbles 2-col + tables) or `mistral` (Mistral OCR `mistral-ocr-latest`, clean markdown incl. tables; caches `<name>.mistral.md` to avoid re-paying). Default parser is still `pypdf` — user wants Mistral going forward (offered to flip `DEFAULT_PARSER`, not yet done).
- `llm.py` — `extract(system, user, backend, model)`; backends: `groq` (default, **`llama-3.3-70b-versatile`**, free, OpenAI-compatible JSON mode temp=0), `ollama` (local, default `llama3.1:8b`, reads `http://localhost:11434/v1`), `anthropic` (`claude-opus-4-8`, paid). Reads `GROQ_API_KEY.txt` fallback.
- `schema.py` — `Record{key,value,evidence,confidence}`, `Extraction{records[]}`.
- `extract.py` — CLI: `--parser --backend --model --no-dedupe --out`. Grounding is whitespace/hyphen-insensitive substring match (survives line-wrap/ligature artifacts). Dedupe collapses exact `(key, lower(value))`, keeps highest confidence — lossy (drops mention frequency + extra evidence); does NOT merge near-dups.
- `render.py` → `view.html` — clustered-by-key HTML view (user found it "useless", prefers JSON; keep but don't prioritize).
- Keys/secrets (gitignored via `*API_KEY.txt`): `GROQ_API_KEY.txt`, `MISTRAL_API_KEY.txt` (Mistral key copied from `../AST/API_KEY.txt`).

**Results so far on ResNet (arXiv 1512.03385, `samples/resnet.pdf`):** pypdf → 295 records (`records.json`, 94% grounded); Mistral OCR → 246 records (`records_mistral.json`, 96% grounded, 114 markdown-table rows). Verdict: **Mistral is better on QUALITY not count** — pypdf's extra records were garbled fragments (`1.8×10^9`→`1.8�10^9`, `conv2_x`→`conv2x`, context-less `output_size: 112`); Mistral captured detection tables (mAP, methods) pypdf missed and preserved units. Record count is a bad quality metric.

**Why (context to resume):** hit **Groq free-tier daily cap (100k tokens/day, per-model)** doing full-paper runs — blocked the pending "no-dedupe raw count" test. Also: local **70B is NOT feasible on the laptop** (RTX 4070 Ti, 12GB VRAM + 34GB RAM → 70B only at Q2 w/ CPU offload, ~1-2 tok/s). That's the whole reason for moving to the GPU server.

**How to apply (resume on GPU server):**
1. Code is pushed to **public** GitHub `Farees-Siddiqui/Masters` under `Structured_Data/` (commit c6777fe). Secrets are NOT in the repo (gitignored) — user chose keep-public + env-vars. On the server: `git pull`, `pip install -r requirements.txt`, then `export GROQ_API_KEY=...` and `export MISTRAL_API_KEY=...` (code reads env first, then local `*_API_KEY.txt`). Cached resnet OCR (`samples/resnet.mistral.md`) is committed so the demo runs without a Mistral key.
2. On the GPU array, run **local 70B via Ollama** (no daily caps): `ollama pull llama3.3:70b`, then finish the pending test — `python extract.py samples/resnet.pdf --parser mistral --no-dedupe --backend ollama --model llama3.3:70b --out records_mistral_raw.json` — to get the raw (pre-dedupe) record count.
3. Then next stages: (a) **key canonicalization** — embed the ~28 emergent keys, cluster near-dups (`metric`↔`result`, `flops`↔`model_size`, hyperparam singletons lr/momentum/batch/weight_decay→`hyperparameter`); (b) **section-scoping** so the references section stops dominating (references bled in as ~54 `author`/53 `result`/14 `publication`); (c) **less-lossy dedupe** (keep `count` + evidence list); (d) tiny **gold-set + precision/recall eval** (user annotated tools before — see [[benchmark-v2-and-annotation-tool]]).
