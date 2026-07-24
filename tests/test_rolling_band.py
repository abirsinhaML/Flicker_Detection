"""Unit tests for rolling-band measurements."""

from __future__ import annotations

import unittest

import numpy as np

from src.core.types import SignalFeatures
from src.detectors.rolling_band import RollingBandDetector
from src.features.extractor import FeatureExtractor


class RollingBandDetectorTests(unittest.TestCase):
    """Exercise the row-profile detector without video decoding."""

    fps = 30.0

    def setUp(self) -> None:
        self.detector = RollingBandDetector(smoothing_sigma=2.0)

    def detect(self, profiles: np.ndarray):
        signals = SignalFeatures(
            luma=np.zeros(profiles.shape[0], dtype=np.float32),
            chroma_a=np.zeros(profiles.shape[0], dtype=np.float32),
            chroma_b=np.zeros(profiles.shape[0], dtype=np.float32),
            row_profiles=profiles,
        )
        return self.detector.detect(FeatureExtractor.extract(signals, self.fps))

    def test_descending_horizontal_edge_has_strength_and_velocity(self) -> None:
        frame_count = 30
        height = 80
        profiles = np.full((frame_count, height), 100.0, dtype=np.float32)
        for frame_index in range(frame_count):
            profiles[frame_index, 15 + frame_index :] += 60.0

        metrics = self.detect(profiles)

        self.assertGreater(metrics.edge_energy, 0.0)
        self.assertGreater(metrics.dominant_band_strength, 0.0)
        self.assertGreater(metrics.vertical_velocity, 0.0)
        self.assertGreater(metrics.position_variance, 0.0)

    def test_uniform_profiles_have_no_band_measurements(self) -> None:
        profiles = np.full((30, 80), 100.0, dtype=np.float32)

        metrics = self.detect(profiles)

        self.assertEqual(metrics.edge_energy, 0.0)
        self.assertEqual(metrics.dominant_band_strength, 0.0)
        self.assertEqual(metrics.vertical_velocity, 0.0)
        self.assertEqual(metrics.position_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
