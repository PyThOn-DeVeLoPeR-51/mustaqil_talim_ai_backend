from __future__ import annotations

import unittest

import numpy as np

from app.ai.common.hough import normalize_hough_lines


class HoughCompatibilityTestCase(unittest.TestCase):
    def test_normalizes_opencv_n_1_4_shape(self) -> None:
        lines = np.asarray([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int32)
        normalized = normalize_hough_lines(lines)
        self.assertEqual(normalized.shape, (2, 4))
        self.assertEqual(normalized.tolist(), [[1, 2, 3, 4], [5, 6, 7, 8]])

    def test_accepts_n_4_shape(self) -> None:
        lines = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
        normalized = normalize_hough_lines(lines)
        self.assertEqual(normalized.shape, (2, 4))

    def test_accepts_single_line_flat_shape(self) -> None:
        lines = np.asarray([1, 2, 3, 4], dtype=np.int32)
        normalized = normalize_hough_lines(lines)
        self.assertEqual(normalized.shape, (1, 4))
        self.assertEqual(normalized[0].tolist(), [1, 2, 3, 4])

    def test_none_returns_empty_matrix(self) -> None:
        normalized = normalize_hough_lines(None)
        self.assertEqual(normalized.shape, (0, 4))

    def test_invalid_shape_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_hough_lines(np.asarray([1, 2, 3], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
