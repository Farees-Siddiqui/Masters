# Incident Scene Analyzer — SAM 3

A demo web app that takes an incident/accident photo, runs **SAM 3** promptable
concept segmentation over an editable open-vocabulary of scene objects, labels
every instance, and infers the likely incident type. Built to showcase
multimodal (vision + language) scene understanding.

## Stack
- **SAM 3** (`facebook/sam3`) via 🤗 `transformers` v5 — Promptable Concept Segmentation
- **Qwen2.5-VL-7B-Instruct** (4-bit, `bitsandbytes`) — on-device incident-narrative VLM
- **FastAPI** backend, vanilla JS frontend
- GPU: uses CUDA if available (tested target: RTX 4070 Ti, 12 GB). Both models
  resident at ~9.4 GB VRAM. Everything runs locally — no image data leaves the machine.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
# CUDA build of torch first:
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install -r requirements.txt
```

### Model access
`facebook/sam3` may be gated on Hugging Face. If the first run fails to download:
```powershell
.venv\Scripts\python -m pip install -U "huggingface_hub[cli]"
.venv\Scripts\huggingface-cli login   # accept the model license on its HF page first
```

## Run
```powershell
.venv\Scripts\python -m uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000

## How it works (multimodal fusion)
1. `app/sam3_engine.py` — loads SAM 3 once, loops the concept list (SAM 3 takes one
   text concept at a time), aggregates masks/boxes/scores, renders the overlay.
2. `app/vlm.py` — Qwen2.5-VL-7B (4-bit) reads the **image** *plus* the grounded SAM 3
   detections and writes a structured **incident report** (accident_type, severity,
   narrative, immediate_actions, key_evidence). Runs on-device.
3. `app/accident.py` — fast transparent heuristic (fallback / headline chip) that
   scores incident archetypes from the detected labels.
4. `app/main.py` — FastAPI endpoints (`/`, `/config`, `/analyze`). `narrative=true`
   form flag toggles the VLM; on VLM error it falls back to the heuristic.

The story: SAM 3 *grounds* the objects, the VLM *reasons* over pixels + grounding.
The VLM gets things the detector can't — e.g. on the sample it reports "Collision
with tree", not just "car + debris".

## Testing
- `smoke_test.py <img>` — SAM 3 only, saves an annotated overlay.
- `vlm_test.py <img>` — SAM 3 + VLM end-to-end, prints the incident report + VRAM.

## Next steps
- Batch SAM 3 concepts in a single forward pass if throughput matters.
- Stream the report token-by-token to the UI; two-phase spinner (segment → narrate).
- Interactive canvas (click a mask to inspect) for live demos.
- Optional cloud-VLM toggle for maximum narrative polish on hero screenshots.
