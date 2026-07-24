"""Unit tests for illuminant-flicker evidence scoring."""

from __future__ import annotations

import unittest

import numpy as np

from src.core.types import SignalFeatures
from src.detectors.illuminant import IlluminantDetector
from src.features.extractor import FeatureExtractor


class IlluminantDetectorTests(unittest.TestCase):
    """Exercise detection using signals only, independently of video I/O."""

    fps = 30.0

    def setUp(self) -> None:
        self.detector = IlluminantDetector(min_prominence=4.0, min_ratio=8.0)
        self.timestamps = np.arange(300, dtype=np.float32) / self.fps

    def detect(self, signal: np.ndarray):
        signals = SignalFeatures(
            luma=signal,
            chroma_a=np.zeros_like(signal),
            chroma_b=np.zeros_like(signal),
            row_profiles=np.zeros((signal.size, 4), dtype=np.float32),
        )
        return self.detector.detect(FeatureExtractor.extract(signals, self.fps))

    def test_flat_signal_scores_zero(self) -> None:
        result = self.detect(np.full(300, 100.0, dtype=np.float32))

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.confidence, 0.0)

    def test_strong_periodic_signal_scores_high(self) -> None:
        signal = 100.0 + 8.0 * np.sin(2.0 * np.pi * 7.0 * self.timestamps)
        result = self.detect(signal)

        self.assertEqual(result.score, 1.0)
        self.assertAlmostEqual(result.dominant_frequency, 7.0, delta=0.15)
        self.assertGreater(result.peak_prominence, 4.0)
        self.assertGreater(result.peak_to_median_ratio, 8.0)

    def test_random_noise_scores_low(self) -> None:
        signal = np.random.default_rng(7).normal(100.0, 1.0, 300)
        result = self.detect(signal)

        self.assertLessEqual(result.score, 0.5)

    def test_linear_exposure_drift_scores_zero_after_detrending(self) -> None:
        signal = 100.0 + 2.0 * self.timestamps
        result = self.detect(signal)

        self.assertEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
