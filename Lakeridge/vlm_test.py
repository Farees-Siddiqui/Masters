"""End-to-end test: SAM 3 segmentation + Qwen2.5-VL incident report, both on GPU."""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import json
import torch
from PIL import Image

from app.sam3_engine import engine as sam
from app.vlm import engine as vlm

img_path = sys.argv[1] if len(sys.argv) > 1 else "samples/crash1.jpg"
concepts = ["car", "person", "wheel", "debris", "broken glass"]
image = Image.open(img_path).convert("RGB")

print(f"[SAM 3] segmenting {img_path} ...")
dets, t_seg = sam.analyze(image, concepts, threshold=0.5)
print(f"  -> {len(dets)} instances in {t_seg:.2f}s")
print(f"  VRAM after SAM3: {torch.cuda.memory_allocated()/1e9:.1f} GB")

detections = [{"label": d.label} for d in dets]
print("[VLM] loading Qwen2.5-VL-7B (4-bit) + generating report ...")
report, t_vlm = vlm.describe_incident(image, detections)
print(f"  -> report in {t_vlm:.1f}s")
print(f"  VRAM after VLM:  {torch.cuda.memory_allocated()/1e9:.1f} GB / "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB total")
print("\n===== INCIDENT REPORT =====")
print(json.dumps(report, indent=2, ensure_ascii=False))
