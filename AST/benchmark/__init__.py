"""Alignment benchmarking.

Two modes share one scoring core (:mod:`benchmark.metrics`):

* **synthetic** (:mod:`benchmark.synthetic`) — no download. Real AST text is split
  into "boxes" and corrupted with controllable OCR-style noise; the true
  ``node -> box`` mapping is known by construction, so we can measure
  precision/recall/F1 as a function of noise. This is the controlled robustness
  test: semantic ``similarity`` should hold up as noise rises while the exact
  ``stream`` matcher degrades.

* **docbank** (planned) — real arXiv pages from DocBank. Ground-truth
  ``node <-> box`` correspondence comes from bounding-box IoU (geometry is the
  oracle), with real PaddleOCR noise on the PDF side.

Run with ``python -m benchmark`` (see :mod:`benchmark.__main__`).
"""
