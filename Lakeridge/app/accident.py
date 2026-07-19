"""Heuristic accident/incident inference from the set of segmented concepts.

This is intentionally simple and transparent: given the labels SAM 3 found in the
scene, we score a handful of incident archetypes. It is a demo-grade reasoning
layer. The production path is to replace / augment this with a vision-language
model that reads the pixels directly (see README).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Archetype:
    label: str
    # concepts that push toward this archetype, with a weight
    signals: dict[str, float]
    # human-readable template; {objs} filled with the matched objects
    rationale: str
    icon: str = "⚠️"
    # at least one of these groups must be present for the archetype to fire
    requires_any: list[list[str]] = field(default_factory=list)


VEHICLES = ["car", "truck", "bus", "motorcycle", "bicycle", "van"]

ARCHETYPES: list[Archetype] = [
    Archetype(
        label="Vehicle collision / traffic accident",
        icon="🚗",
        signals={
            **{v: 1.0 for v in VEHICLES},
            "person": 0.6,
            "debris": 0.8,
            "traffic cone": 0.4,
            "traffic light": 0.2,
            "broken glass": 0.7,
        },
        requires_any=[VEHICLES],
        rationale="Detected {objs} in the scene, consistent with a roadway collision involving one or more vehicles.",
    ),
    Archetype(
        label="Fire / smoke incident",
        icon="🔥",
        signals={"fire": 1.5, "smoke": 1.2, "person": 0.3},
        requires_any=[["fire", "smoke"]],
        rationale="Presence of {objs} indicates an active fire or smoke event requiring evacuation and suppression.",
    ),
    Archetype(
        label="Fall from height",
        icon="🪜",
        signals={"ladder": 1.2, "scaffolding": 1.2, "person": 0.8, "fallen person": 1.5},
        requires_any=[["ladder", "scaffolding", "fallen person"]],
        rationale="Detected {objs}, a pattern associated with a fall-from-height injury.",
    ),
    Archetype(
        label="Slip / trip hazard",
        icon="💧",
        signals={"spilled liquid": 1.3, "wet floor sign": 1.0, "fallen person": 1.2, "person": 0.5},
        requires_any=[["spilled liquid", "wet floor sign", "fallen person"]],
        rationale="Detected {objs}, indicating a slip/trip hazard on the walking surface.",
    ),
    Archetype(
        label="Industrial / material handling incident",
        icon="🏭",
        signals={"forklift": 1.3, "pallet": 0.8, "debris": 0.6, "person": 0.6, "helmet": 0.3},
        requires_any=[["forklift", "pallet"]],
        rationale="Detected {objs}, consistent with a material-handling or warehouse incident.",
    ),
]


def infer_accident(labels: Iterable[str]) -> dict:
    """Return the most likely incident archetype for the detected labels."""
    present = {l.lower() for l in labels}
    scored: list[tuple[float, Archetype, list[str]]] = []

    for arc in ARCHETYPES:
        # gate: at least one required group must have a member present
        if arc.requires_any and not any(
            any(member in present for member in group) for group in arc.requires_any
        ):
            continue
        matched = [c for c in arc.signals if c in present]
        if not matched:
            continue
        score = sum(arc.signals[c] for c in matched)
        scored.append((score, arc, matched))

    if not scored:
        return {
            "label": "Undetermined incident",
            "icon": "❓",
            "confidence": 0.0,
            "rationale": "No strong incident signals were detected among the segmented objects. "
            "Try adding more specific concepts to the prompt.",
        }

    scored.sort(key=lambda t: t[0], reverse=True)
    top_score, arc, matched = scored[0]
    total = sum(s for s, _, _ in scored)
    confidence = top_score / total if total else 0.0

    objs = ", ".join(sorted(set(matched)))
    return {
        "label": arc.label,
        "icon": arc.icon,
        "confidence": round(confidence, 3),
        "rationale": arc.rationale.format(objs=objs),
    }
