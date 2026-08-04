"""Unit tests for frequency-selective rolling-band measurements."""

from __future__ import annotations

import unittest

import numpy as np

from src.core.types import SignalFeatures
from src.detectors.rolling_band import RollingBandDetector
from src.features.extractor import FeatureExtractor


class RollingBandDetectorTests(unittest.TestCase):
    """Exercise the row-profile detector without video decoding."""

    fps = 30.0
    flicker_hz = 6.0

    def setUp(self) -> None:
        self.detector = RollingBandDetector(smoothing_sigma=2.0)

    def detect(
        self,
        profiles: np.ndarray,
        column_bands: np.ndarray | None = None,
    ):
        """Run the detector, supplying a luma channel that carries the flicker.

        Global luma is what fixes the frequency every measurement is taken at, so
        it is set explicitly rather than derived from the profiles: a band with a
        whole number of cycles across the frame averages out of the row mean and
        would leave no frequency to project onto.
        """
        frames = profiles.shape[0]
        times = np.arange(frames, dtype=np.float32) / self.fps
        luma = 100.0 + 5.0 * np.cos(2.0 * np.pi * self.flicker_hz * times)
        if column_bands is None:
            column_bands = np.repeat(profiles[None, ...], 3, axis=0)
        signals = SignalFeatures(
            luma=luma.astype(np.float32),
            chroma_a=np.zeros(frames, dtype=np.float32),
            chroma_b=np.zeros(frames, dtype=np.float32),
            row_profiles=profiles,
            column_band_profiles=column_bands,
        )
        return self.detector.detect(FeatureExtractor.extract(signals, self.fps))

    def _band(
        self,
        cycles_per_frame: float = 1.0,
        depth: float = 0.10,
        frames: int = 90,
        height: int = 120,
        downward: bool = True,
        level: float = 100.0,
        phase: float = 0.0,
    ) -> np.ndarray:
        """A sinusoidal band of ``depth`` relative swing drifting down the rows.

        Negative spatial frequency gives downward motion: constant phase requires
        ``f*t + k*r`` fixed, so ``dr/dt = -f/k`` is positive when ``k < 0``.
        """
        rows = np.arange(height)[None, :]
        times = np.arange(frames)[:, None] / self.fps
        spatial = (-1.0 if downward else 1.0) * cycles_per_frame / height
        argument = 2.0 * np.pi * (self.flicker_hz * times + spatial * rows) + phase
        return (level * (1.0 + depth * np.cos(argument))).astype(np.float32)

    def test_measures_the_relative_swing_of_a_band(self) -> None:
        metrics = self.detect(self._band(depth=0.10))

        self.assertAlmostEqual(metrics.band_amplitude, 0.10, delta=0.015)

    def test_band_amplitude_is_scale_free(self) -> None:
        """A 10% swing reads the same in dim and bright footage."""
        dim = self.detect(self._band(depth=0.10, level=20.0)).band_amplitude
        bright = self.detect(self._band(depth=0.10, level=200.0)).band_amplitude

        self.assertAlmostEqual(dim, bright, delta=0.01)

    def test_a_coherent_band_has_linear_per_row_phase(self) -> None:
        metrics = self.detect(self._band())

        self.assertGreater(metrics.phase_linearity, 0.95)
        self.assertGreater(metrics.horizontal_coherence, 0.95)

    def test_recovers_the_number_of_bands_across_the_frame(self) -> None:
        for cycles in (0.5, 1.0, 2.0):
            with self.subTest(cycles=cycles):
                metrics = self.detect(self._band(cycles_per_frame=cycles))
                self.assertAlmostEqual(metrics.band_cycles_per_frame, cycles, delta=0.2)

    def test_drift_direction_is_signed(self) -> None:
        down = self.detect(self._band(downward=True)).drift_velocity
        up = self.detect(self._band(downward=False)).drift_velocity

        self.assertGreater(down, 0.0)
        self.assertLess(up, 0.0)
        self.assertAlmostEqual(down, -up, delta=abs(down) * 0.1)

    def test_spatially_uniform_flicker_is_not_this_detectors_business(self) -> None:
        """Every row moving in lockstep is global flicker, not banding.

        It is real flicker and the illuminant detector should score it; this one
        reports only the spatial structure a sequential readout leaves behind.
        """
        uniform = self.detect(self._band(cycles_per_frame=0.0))
        banded = self.detect(self._band(cycles_per_frame=1.0))

        self.assertLess(uniform.band_amplitude, 0.005)
        self.assertLess(uniform.band_amplitude, banded.band_amplitude / 10.0)

    def test_broadband_motion_is_not_read_as_banding(self) -> None:
        """The regression the previous detector failed.

        Vigorous motion produces large row-profile swings with no periodic
        structure at the flicker frequency, and must not look like a band.
        """
        rng = np.random.default_rng(7)
        frames, height = 90, 120
        walk = np.cumsum(rng.normal(0.0, 6.0, (frames, 1)), axis=0)
        texture = rng.normal(0.0, 12.0, (1, height))
        motion = (100.0 + walk + texture + rng.normal(0.0, 4.0, (frames, height))).astype(
            np.float32
        )

        moving = self.detect(motion)
        banded = self.detect(self._band(depth=0.10))

        self.assertLess(moving.band_amplitude, banded.band_amplitude / 3.0)

    def test_randomised_per_row_phase_destroys_linearity(self) -> None:
        """Same swing, unrelated phases: an oscillation but not a band."""
        rng = np.random.default_rng(3)
        frames, height = 90, 120
        times = np.arange(frames)[:, None] / self.fps
        phases = rng.uniform(0.0, 2.0 * np.pi, (1, height))
        scrambled = (
            100.0 * (1.0 + 0.10 * np.cos(2.0 * np.pi * self.flicker_hz * times + phases))
        ).astype(np.float32)

        coherent = self.detect(self._band(depth=0.10))
        incoherent = self.detect(scrambled)

        self.assertLess(incoherent.phase_linearity, 0.6)
        self.assertLess(incoherent.phase_linearity, coherent.phase_linearity / 1.6)

    def test_uniform_profiles_have_no_band_measurements(self) -> None:
        metrics = self.detect(np.full((30, 80), 100.0, dtype=np.float32))

        self.assertEqual(metrics.band_amplitude, 0.0)
        self.assertEqual(metrics.band_cycles_per_frame, 0.0)
        self.assertEqual(metrics.drift_velocity, 0.0)

    def test_a_band_bent_across_the_width_loses_coherence(self) -> None:
        """What a dewarped or stabilized frame looks like: row-dependent phase."""
        profiles = self._band()
        bent = np.stack([self._band(phase=p) for p in (0.0, 2.0, 4.0)])

        straight = self.detect(profiles).horizontal_coherence
        remapped = self.detect(profiles, column_bands=bent).horizontal_coherence

        self.assertLess(remapped, straight)

    def test_coherence_defaults_to_trusting_the_model_without_evidence(self) -> None:
        empty = np.empty((0, 90, 120), dtype=np.float32)
        metrics = self.detect(self._band(), column_bands=empty)

        self.assertEqual(metrics.horizontal_coherence, 1.0)

    def test_rejects_malformed_profiles(self) -> None:
        with self.assertRaises(ValueError):
            self.detect(np.zeros((4, 4, 4), dtype=np.float32))

    def test_rejects_non_positive_smoothing(self) -> None:
        with self.assertRaises(ValueError):
            RollingBandDetector(smoothing_sigma=0.0)


if __name__ == "__main__":
    unittest.main()
