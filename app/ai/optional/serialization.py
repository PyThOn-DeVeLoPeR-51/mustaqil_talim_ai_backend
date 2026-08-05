"""JSON serialization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

def _json_default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


__all__ = [
    '_json_default',
]
