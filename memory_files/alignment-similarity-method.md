---
name: alignment-similarity-method
description: similarity_aligner.py — semantic embedding cosine aligner; /api/align/compute?method=similarity|stream
metadata: 
  node_type: memory
  type: project
  originSessionId: fbb2a4fa-a5e3-4f1a-a503-0c9f7a785305
---

`alignment/similarity_aligner.py` (`align_similarity`) is the semantic alignment method added alongside `align_stream`. It embeds AST node text and PDF box text with a sentence-transformer (`all-MiniLM-L6-v2`, lazy-loaded, process-cached, `_device()` auto-uses CUDA if a CUDA torch is present — currently torch is CPU-only, fine for MiniLM) and matches by cosine similarity ≥ threshold (default 0.5). Same return shape as `align_stream` (`alignment`/`reverse`/`coverage`).

Wired switchably: `GET /api/align/compute?doc=&granularity=&method=stream|similarity&threshold=`, dispatched via `ALIGN_METHODS` in `app/main.py`. Frontend has a Method dropdown (`static/align.html`/`align.js`); result cache keyed by `granularity|method`. Dep: `sentence-transformers` in requirements.txt.

**Strengths/weaknesses** (quantified in [[alignment-benchmark]]): order-invariant and robust to OCR noise that breaks `stream`'s positional char-diff; best at paragraph granularity (node text ≈ box text), weak at line/word fragments. On real resnet.pdf paragraph: similarity 97.7% coverage vs stream 94.5%. Recall is throttled by the fixed cosine threshold under heavy noise (a `--threshold` sweep is the planned fix).

Part of [[project-overview]].
