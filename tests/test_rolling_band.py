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

    def detect(self, profiles: np.ndarray, column_bands: np.ndarray | None = None):
        if column_bands is None:
            # Identical slices: the default straight-band case.
            column_bands = np.repeat(profiles[None, ...], 3, axis=0)
        signals = SignalFeatures(
            # Global luma really is the row profiles averaged down, and the
            # coherence check keys off its flicker frequency.
            luma=profiles.mean(axis=1).astype(np.float32),
            chroma_a=np.zeros(profiles.shape[0], dtype=np.float32),
            chroma_b=np.zeros(profiles.shape[0], dtype=np.float32),
            row_profiles=profiles,
            column_band_profiles=column_bands,
        )
        return self.detector.detect(FeatureExtractor.extract(signals, self.fps))

    @staticmethod
    def _descending_edge(frame_count: int = 30, height: int = 80, offset: int = 0) -> np.ndarray:
        profiles = np.full((frame_count, height), 100.0, dtype=np.float32)
        for frame_index in range(frame_count):
            profiles[frame_index, 15 + frame_index + offset :] += 60.0
        return profiles

    def _banded(self, phase: float = 0.0, frames: int = 90, height: int = 80) -> np.ndarray:
        """Row profiles carrying a periodic band drifting down the frame.

        Half a band cycle spans the frame, matching what the corpus actually
        shows: a single top-to-bottom gradient. A whole number of cycles would
        average out of the global luma signal and leave no flicker frequency.
        """
        rows = np.arange(height)[None, :]
        times = np.arange(frames)[:, None] / self.fps
        offset = rows / (2.0 * height)
        return (100.0 + 20.0 * np.sin(2.0 * np.pi * (6.0 * times + offset) + phase)).astype(
            np.float32
        )

    def test_descending_horizontal_edge_has_strength_and_velocity(self) -> None:
        metrics = self.detect(self._descending_edge())

        self.assertGreater(metrics.edge_energy, 0.0)
        self.assertGreater(metrics.dominant_band_strength, 0.0)
        self.assertGreater(metrics.vertical_velocity, 0.0)
        self.assertGreater(metrics.position_variance, 0.0)

    def test_a_band_crossing_the_full_width_in_phase_is_coherent(self) -> None:
        profiles = self._banded()
        metrics = self.detect(profiles, column_bands=np.repeat(profiles[None], 3, axis=0))

        self.assertGreater(metrics.horizontal_coherence, 0.95)

    def test_a_band_bent_across_the_width_loses_coherence(self) -> None:
        """What a dewarped or stabilized frame looks like: row-dependent phase."""
        profiles = self._banded()
        bent = np.stack([self._banded(phase=p) for p in (0.0, 2.0, 4.0)])

        straight = self.detect(profiles, column_bands=np.repeat(profiles[None], 3, axis=0))
        remapped = self.detect(profiles, column_bands=bent)

        self.assertLess(remapped.horizontal_coherence, straight.horizontal_coherence)

    def test_broadband_motion_does_not_destroy_coherence(self) -> None:
        """The reason the measure is frequency-selective.

        A fisheye turning with the wearer's head produces per-row change that
        differs wildly left to right. Measured broadband, that reads as a bent
        band; measured at the flicker frequency, the banding still agrees.
        """
        rng = np.random.default_rng(11)
        profiles = self._banded()
        noisy = np.stack(
            [profiles + rng.normal(0.0, 15.0, profiles.shape).astype(np.float32) for _ in range(3)]
        )
        metrics = self.detect(profiles, column_bands=noisy)

        self.assertGreater(metrics.horizontal_coherence, 0.8)

    def test_coherence_defaults_to_trusting_the_model_without_evidence(self) -> None:
        empty = np.empty((0, 30, 80), dtype=np.float32)
        self.assertEqual(self.detect(self._banded(), column_bands=empty).horizontal_coherence, 1.0)
        # A flat, aperiodic signal has no band whose straightness to judge.
        flat = np.full((30, 80), 100.0, dtype=np.float32)
        self.assertEqual(self.detect(flat).horizontal_coherence, 1.0)

    def test_uniform_profiles_have_no_band_measurements(self) -> None:
        profiles = np.full((30, 80), 100.0, dtype=np.float32)

        metrics = self.detect(profiles)

        self.assertEqual(metrics.edge_energy, 0.0)
        self.assertEqual(metrics.dominant_band_strength, 0.0)
        self.assertEqual(metrics.vertical_velocity, 0.0)
        self.assertEqual(metrics.position_variance, 0.0)


if __name__ == "__main__":
    unittest.main()
