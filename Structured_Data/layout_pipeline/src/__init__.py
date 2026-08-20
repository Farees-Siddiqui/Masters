"""Document extraction + reading-order recovery.

PaddleOCR for detection/recognition, XY-Cut++ (arXiv:2504.10258) for reading
order.
"""

from .xycut_plus import (DEFAULT_CONFIG, XYCutConfig, compute_reading_order,
                         order_indices)

__all__ = [
    "XYCutConfig",
    "DEFAULT_CONFIG",
    "compute_reading_order",
    "order_indices",
]
