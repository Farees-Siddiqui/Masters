"""Vision-Language incident-narrative layer.

Runs Qwen2.5-VL-7B-Instruct (4-bit, on-GPU) to read the incident image directly
and produce a structured incident report. The SAM 3 detections are passed in as
grounding context, so the narrative is fused from *both* the pixels and the
grounded object segmentation — that is the multimodal story.

Everything runs locally; no image data leaves the machine.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Cap visual tokens to keep VRAM in check on the 12GB card (SAM 3 is resident too).
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

SYSTEM_PROMPT = (
    "You are an expert incident-scene analyst supporting emergency and safety teams. "
    "You examine a photograph of a potential accident or incident and produce a precise, "
    "sober assessment. You never speculate beyond visual evidence. You are concise and factual."
)

USER_TEMPLATE = """Analyze this incident photograph.

A segmentation model has already grounded these objects in the scene (label × count):
{objects}

Using BOTH the image and the grounded objects, return a single JSON object with EXACTLY these fields:
- "accident_type": short phrase (e.g. "Single-vehicle collision with fixed object")
- "severity": one of "Low", "Moderate", "Severe", "Critical"
- "confidence": number 0-1 for your assessment
- "narrative": 2-3 sentence factual description of what happened and the current scene state
- "immediate_actions": array of 2-4 short recommended first-response actions
- "key_evidence": array of 2-4 short visual cues that support your assessment

Return ONLY the JSON object, no prose, no markdown fences."""


@dataclass
class IncidentReport:
    accident_type: str
    severity: str
    confidence: float
    narrative: str
    immediate_actions: list[str]
    key_evidence: list[str]
    source: str = "vlm"  # "vlm" or "fallback"

    def to_dict(self) -> dict:
        return {
            "accident_type": self.accident_type,
            "severity": self.severity,
            "confidence": round(float(self.confidence), 3),
            "narrative": self.narrative,
            "immediate_actions": self.immediate_actions,
            "key_evidence": self.key_evidence,
            "source": self.source,
        }


class VlmEngine:
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=quant,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )

    @torch.no_grad()
    def describe_incident(self, image: Image.Image, detections: list) -> tuple[dict, float]:
        """detections: list of objects with .label (Detection) or dicts with 'label'."""
        self.load()
        t0 = time.time()

        objects = _summarize_objects(detections)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": USER_TEMPLATE.format(objects=objects)},
                ],
            },
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        generated = self._model.generate(
            **inputs, max_new_tokens=512, do_sample=False, temperature=None, top_p=None
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        raw = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        torch.cuda.empty_cache()
        report = _parse_report(raw)
        return report.to_dict(), time.time() - t0


def _summarize_objects(detections: list) -> str:
    counts: dict[str, int] = {}
    for d in detections:
        label = d["label"] if isinstance(d, dict) else getattr(d, "label", None)
        if label:
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "(none detected)"
    return ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))


def _parse_report(raw: str) -> IncidentReport:
    """Robustly extract the JSON object from the model output."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    data: dict = {}
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = {}

    def _list(v):
        if isinstance(v, list):
            return [str(x) for x in v][:4]
        if isinstance(v, str) and v:
            return [v]
        return []

    return IncidentReport(
        accident_type=str(data.get("accident_type") or "Unclassified incident"),
        severity=str(data.get("severity") or "Moderate"),
        confidence=_coerce_float(data.get("confidence"), 0.5),
        narrative=str(data.get("narrative") or (raw.strip()[:400] if not data else "")),
        immediate_actions=_list(data.get("immediate_actions")),
        key_evidence=_list(data.get("key_evidence")),
        source="vlm",
    )


def _coerce_float(v, default: float) -> float:
    try:
        f = float(v)
        return max(0.0, min(1.0, f))
    except (TypeError, ValueError):
        return default


engine = VlmEngine()
