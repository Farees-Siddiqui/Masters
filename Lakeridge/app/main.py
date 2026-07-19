"""FastAPI app: incident-image semantic segmentation & labeling with SAM 3."""
from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .accident import infer_accident
from .sam3_engine import (
    DEFAULT_CONCEPTS,
    engine,
    image_to_png_bytes,
    render_overlay,
)
from .vlm import engine as vlm_engine

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Incident Scene Analyzer — SAM 3")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/config")
def config() -> dict:
    return {
        "default_concepts": DEFAULT_CONCEPTS,
        "device": engine.device,
        "model_loaded": engine.loaded,
        "vlm_loaded": vlm_engine.loaded,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    concepts: str = Form(""),
    threshold: float = Form(0.5),
    narrative: bool = Form(True),
) -> JSONResponse:
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    concept_list = [c.strip() for c in concepts.split(",") if c.strip()] or DEFAULT_CONCEPTS

    try:
        dets, elapsed = engine.analyze(pil, concept_list, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    overlay = render_overlay(pil, dets)
    overlay_b64 = base64.b64encode(image_to_png_bytes(overlay)).decode("ascii")

    detections = [
        {
            "label": d.label,
            "score": round(d.score, 4),
            "box": [round(v, 1) for v in d.box],
            "color": "#%02x%02x%02x" % d.color,
        }
        for d in sorted(dets, key=lambda x: x.score, reverse=True)
    ]

    # per-label counts for the summary panel
    counts: dict[str, dict] = {}
    for d in detections:
        c = counts.setdefault(d["label"], {"label": d["label"], "count": 0, "color": d["color"]})
        c["count"] += 1

    accident = infer_accident([d["label"] for d in detections])

    # VLM incident report (reads pixels + grounded detections). Falls back to the
    # heuristic verdict if the VLM is disabled or errors out.
    report = None
    vlm_sec = 0.0
    if narrative:
        try:
            report, vlm_sec = vlm_engine.describe_incident(pil, detections)
        except Exception as exc:  # noqa: BLE001
            report = {
                "accident_type": accident["label"],
                "severity": "Moderate",
                "confidence": accident["confidence"],
                "narrative": accident["rationale"],
                "immediate_actions": [],
                "key_evidence": [],
                "source": "fallback",
                "error": str(exc)[:200],
            }

    return JSONResponse(
        {
            "overlay": f"data:image/png;base64,{overlay_b64}",
            "detections": detections,
            "summary": sorted(counts.values(), key=lambda x: x["count"], reverse=True),
            "accident": accident,
            "report": report,
            "concepts_used": concept_list,
            "elapsed_sec": round(elapsed, 2),
            "vlm_sec": round(vlm_sec, 2),
            "device": engine.device,
        }
    )


# static assets (css/js)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
