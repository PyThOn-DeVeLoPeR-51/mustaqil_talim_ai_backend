from __future__ import annotations

import numpy as np


def normalize_hough_lines(lines) -> np.ndarray:
    """Return OpenCV HoughLinesP output as a stable ``(N, 4)`` int32 array.

    OpenCV builds/wheels can expose HoughLinesP results as ``(N, 1, 4)``,
    ``(N, 4)``, or (for a single segment) a flat ``(4,)`` array.  Drawing AI
    code must not index ``[:, 0]`` blindly because that turns ``(N, 4)`` into
    a 1-D array of scalar ``numpy.int32`` values and tuple-unpacking fails.
    """
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)

    arr = np.asarray(lines)
    if arr.size == 0:
        return np.empty((0, 4), dtype=np.int32)
    if arr.size % 4 != 0:
        raise ValueError(f"Unexpected HoughLinesP output shape: {arr.shape}")
    return arr.reshape(-1, 4).astype(np.int32, copy=False)
