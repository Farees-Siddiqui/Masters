"""Document layout detection (PP-DocLayoutV3).

Renders PDF pages to images and runs PaddleOCR's layout detector, writing
per-page artifacts in a predictable folder structure. Designed to be reused
by both the CLI (`layout.cli`) and, later, the FastAPI web app.
"""

from .detector import ALL_GRANULARITY, GRANULARITIES, LayoutDetector, detect_layout

__all__ = ["LayoutDetector", "detect_layout", "GRANULARITIES", "ALL_GRANULARITY"]
