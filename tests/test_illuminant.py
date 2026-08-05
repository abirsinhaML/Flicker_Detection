"""Unit tests for illuminant-flicker measurement and its normalization."""

from __future__ import annotations

import unittest

import numpy as np

from src.calibration.thresholds import IlluminantNormalizer
from src.core.types import SignalFeatures
from src.detectors.illuminant import IlluminantDetector, IlluminantMetrics
from src.features.extractor import FeatureExtractor


class IlluminantDetectorTests(unittest.TestCase):
    """Exercise measurement using signals only, independently of video I/O."""

    fps = 30.0

    def setUp(self) -> None:
        self.detector = IlluminantDetector()
        self.normalizer = IlluminantNormalizer(
            prominence_reference=4.0,
            periodicity_reference=8.0,
            modulation_depth_reference=0.02,
        )
        self.timestamps = np.arange(300, dtype=np.float32) / self.fps

    def measure(self, signal: np.ndarray) -> IlluminantMetrics:
        signals = SignalFeatures(
            luma=signal,
            chroma_a=np.zeros_like(signal),
            chroma_b=np.zeros_like(signal),
            row_profiles=np.zeros((signal.size, 4), dtype=np.float32),
            column_band_profiles=np.zeros((3, signal.size, 4), dtype=np.float32),
        )
        return self.detector.detect(FeatureExtractor.extract(signals, self.fps))

    def score(self, signal: np.ndarray) -> float:
        return self.normalizer(self.measure(signal))

    def _modulated(self, amplitude: float, frequency: float = 7.0) -> np.ndarray:
        return 100.0 + amplitude * np.sin(2.0 * np.pi * frequency * self.timestamps)

    def test_flat_signal_measures_and_scores_zero(self) -> None:
        metrics = self.measure(np.full(300, 100.0, dtype=np.float32))

        self.assertEqual(metrics.peak_prominence, 0.0)
        self.assertEqual(metrics.peak_to_median_ratio, 0.0)
        self.assertEqual(self.normalizer(metrics), 0.0)

    def test_strong_periodic_signal_scores_high(self) -> None:
        metrics = self.measure(self._modulated(8.0))

        self.assertAlmostEqual(metrics.dominant_frequency, 7.0, delta=0.15)
        self.assertGreater(metrics.peak_prominence, 4.0)
        self.assertGreater(metrics.peak_to_median_ratio, 8.0)
        self.assertGreater(self.normalizer(metrics), 0.5)

    def test_random_noise_scores_low(self) -> None:
        signal = np.random.default_rng(7).normal(100.0, 1.0, 300).astype(np.float32)

        self.assertLess(self.score(signal), 0.5)

    def test_linear_exposure_drift_scores_zero_after_detrending(self) -> None:
        self.assertEqual(self.score(100.0 + 2.0 * self.timestamps), 0.0)

    def test_score_is_strictly_monotone_in_modulation_depth(self) -> None:
        """The property the previous boolean staircase could not provide.

        Three severities used to collapse onto at most three distinct scores,
        leaving nothing for a threshold to separate.
        """
        scores = [self.score(self._modulated(amplitude)) for amplitude in (0.5, 2.0, 8.0, 32.0)]

        self.assertEqual(scores, sorted(scores))
        self.assertEqual(len(set(scores)), len(scores), "distinct severities must not tie")

    def test_strong_artifacts_stay_distinguishable(self) -> None:
        """Well past the reference scale, ordering must survive."""
        strong = self.score(self._modulated(64.0))
        stronger = self.score(self._modulated(256.0))

        self.assertLess(strong, stronger)
        self.assertLess(stronger, 1.0)


if __name__ == "__main__":
    unittest.main()
