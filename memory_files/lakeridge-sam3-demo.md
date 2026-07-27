---
name: lakeridge-sam3-demo
description: Lakeridge project — SAM 3 incident-image segmentation web demo for hospital funding pitch
metadata: 
  node_type: memory
  type: project
  originSessionId: 22ce3e7e-3dcb-43a3-80df-fbb890d3cdbb
---

`Documents/Masters/Lakeridge` is a demo built to secure funding from a nearby hospital. Goal: show **multimodal** scene understanding by analyzing an incident/accident photo — segment + label every object and infer the accident type. Non-medical images are fine for the demo (traffic crashes etc.). Screenshots are the deliverable to "hook" the hospital.

**Stack:** FastAPI + vanilla JS (dark UI), `.venv` in the project. Uses **SAM 3** (`facebook/sam3`, 840M, Promptable Concept Segmentation) via `transformers` v5 on the RTX 4070 Ti (CUDA 12.6). App loops the model over an editable concept vocabulary (SAM 3 takes one text concept at a time), aggregates masks/boxes/scores, renders a colored labeled overlay.

**VLM layer (added 2026-07-04):** `app/vlm.py` runs **Qwen2.5-VL-7B-Instruct in 4-bit** (bitsandbytes nf4, bf16 compute) locally, reads image + SAM 3 detections, returns a structured incident report (accident_type, severity, narrative, immediate_actions, key_evidence) as JSON. User chose local (on-prem privacy story for hospital) over cloud API. Both models resident ≈ 9.4/12.9 GB VRAM — fits. `bitsandbytes 0.49.2` works on Windows CUDA. max_pixels capped to ~1MP to control VRAM. `/analyze` has `narrative=true` flag; VLM errors fall back to `app/accident.py` heuristic (collision/fire/fall/slip/industrial). On the sample crash image the VLM said "Collision with tree" — beats detector-only. Test: `vlm_test.py <img>`.

**Key facts:**
- `facebook/sam3` is a **gated** HF repo — needed Meta manual approval (granted 2026-07-04). Check status: `huggingface_hub.auth_check('facebook/sam3')`. Both the transformers path and the native `facebookresearch/sam3` repo pull the same gated weights.
- Native `sam3` repo also cloned + `pip install -e ./sam3`; on Windows it needs `triton-windows`, `einops`, `pycocotools` (not in its pyproject).
- Run: `.venv\Scripts\python -m uvicorn app.main:app --port 8000`. Endpoints: `/`, `/config`, `/analyze`. `smoke_test.py` for CLI test; `samples/crash1.jpg` is a public-domain test image.
- Windows console is cp1252 — set `PYTHONUTF8=1` when printing emoji (verdict icons).

**Next-step upgrade noted in README:** replace the heuristic accident layer with a real vision-language model that reads pixels for scene description / accident classification. Related: [[project-overview]] (AST project uses same FastAPI+static pattern).
