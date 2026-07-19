"""End-to-end smoke test of the SAM 3 engine + accident layer on a sample image.

Run: .venv\\Scripts\\python smoke_test.py samples/crash1.jpg
Writes samples/<name>_overlay.png and prints detections + verdict.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252
from PIL import Image

from app.sam3_engine import engine, render_overlay
from app.accident import infer_accident

img_path = sys.argv[1] if len(sys.argv) > 1 else "samples/crash1.jpg"
concepts = ["car", "truck", "person", "debris", "wheel", "traffic cone", "broken glass"]

print(f"[1/4] loading image {img_path}")
image = Image.open(img_path).convert("RGB")

print(f"[2/4] running SAM 3 over {len(concepts)} concepts on {engine.device} ...")
dets, elapsed = engine.analyze(image, concepts, threshold=0.5)
print(f"      -> {len(dets)} instances in {elapsed:.2f}s")

print("[3/4] detections:")
for d in sorted(dets, key=lambda x: x.score, reverse=True):
    print(f"      {d.label:16s} {d.score:5.1%}  box={[round(v) for v in d.box]}")

verdict = infer_accident([d.label for d in dets])
print(f"[4/4] VERDICT: {verdict['icon']} {verdict['label']}  ({verdict['confidence']:.0%})")
print(f"      {verdict['rationale']}")

out = img_path.rsplit(".", 1)[0] + "_overlay.png"
render_overlay(image, dets).save(out)
print(f"\nsaved overlay -> {out}")
