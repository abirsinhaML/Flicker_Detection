"""Measurement of rolling horizontal bands from luminance row profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.core.types import FeatureVector
from src.detectors.base import BaseDetector
from src.features.frequency import FrequencyAnalyzer


@dataclass(slots=True)
class RollingBandMetrics:
    """Measurements characterizing horizontal bands and their vertical motion.

    ``horizontal_coherence`` is a self-check on the band model rather than
    evidence of flicker: it reports how equally the frame's vertical slices agree
    on the per-row brightness change.
    """

    edge_energy: float
    dominant_band_strength: float
    vertical_velocity: float
    position_variance: float
    temporal_periodicity: float
    horizontal_coherence: float = 1.0


class RollingBandDetector(BaseDetector):
    """Measure moving horizontal band edges in a ``(frame, row)`` matrix.

    ``vertical_velocity`` is signed and expressed in rows per second.  A
    positive value denotes downward motion in image coordinates.
    """

    def __init__(self, smoothing_sigma: float) -> None:
        if not isfinite(smoothing_sigma) or smoothing_sigma <= 0:
            raise ValueError("smoothing_sigma must be a finite value greater than zero")

        self.smoothing_sigma = smoothing_sigma

    def detect(self, features: FeatureVector) -> RollingBandMetrics:
        """Extract edge, motion, and periodicity measurements from row profiles."""
        profiles = self._validate_profiles(features.row_profiles)
        if profiles.shape[0] == 0 or profiles.shape[1] < 2:
            FrequencyAnalyzer.compute(np.empty(0, dtype=np.float32), features.fps)
            return self._empty_metrics()

        smoothed_profiles = gaussian_filter1d(
            profiles,
            sigma=self.smoothing_sigma,
            axis=1,
        )
        relative_profiles = smoothed_profiles - smoothed_profiles.mean(
            axis=1,
            keepdims=True,
        )
        gradient = np.diff(relative_profiles, axis=1)
        absolute_gradient = np.abs(gradient)
        frame_energy = np.mean(gradient**2, axis=1)
        positions = np.argmax(absolute_gradient, axis=1).astype(np.float32)
        frequency = FrequencyAnalyzer.compute(frame_energy, features.fps)

        return RollingBandMetrics(
            edge_energy=float(np.mean(frame_energy)),
            dominant_band_strength=float(np.mean(np.max(absolute_gradient, axis=1))),
            vertical_velocity=float(np.mean(np.diff(positions)) * features.fps)
            if positions.size > 1
            else 0.0,
            position_variance=float(np.var(positions)),
            temporal_periodicity=frequency.peak_to_median_ratio,
            # The luminance flicker frequency, not this detector's edge-energy
            # frequency: edge energy oscillates at a harmonic of the band drift,
            # and projecting onto it would land on a bin holding no band energy.
            horizontal_coherence=self._horizontal_coherence(
                features.column_band_profiles,
                features.fps,
                max(features.frequency.dominant_frequency, 0.0),
            ),
        )

    @staticmethod
    def _horizontal_coherence(
        column_band_profiles: np.ndarray,
        fps: float,
        flicker_frequency: float,
    ) -> float:
        """Return how consistently vertical slices see the same banding per row.

        A rolling shutter exposes whole sensor rows in sequence, so its banding is
        imposed after the lens has projected the scene.  Fisheye distortion
        therefore bends scene content but leaves the bands straight, and every
        slice of the frame should agree on the brightness of each row.

        The comparison is made only at the flicker frequency.  A raw
        frame-to-frame difference is dominated by scene motion, which under a
        fisheye is wildly different at the left and right edges during head
        rotation -- measuring that would report low coherence for every moving
        video regardless of its banding.  Projecting each row's time series onto
        the dominant frequency keeps the periodic component and rejects the
        broadband motion around it.

        Low coherence with real periodic flicker present means the frame was
        geometrically remapped -- dewarped, rectilinearized, or electronically
        stabilized -- which bends the bands and invalidates this detector's
        model.  Reporting it turns a silent underdetection into a visible number.
        """
        profiles = np.asarray(column_band_profiles, dtype=np.float32)
        if profiles.ndim != 3 or profiles.shape[0] < 2 or profiles.shape[1] < 2:
            return 1.0
        if flicker_frequency <= 0.0 or not isfinite(fps) or fps <= 0.0:
            # No periodic component: there is no band whose shape to check.
            return 1.0

        # Complex amplitude of each row's oscillation at the flicker frequency.
        # Phase matters as much as magnitude: a bent band shows a row-dependent
        # phase shift between slices even when the amplitudes agree.
        frames = profiles.shape[1]
        times = np.arange(frames, dtype=np.float64) / fps
        kernel = np.exp(-2j * np.pi * flicker_frequency * times)
        centered = profiles - profiles.mean(axis=1, keepdims=True)
        amplitudes = np.tensordot(centered.astype(np.float64), kernel, axes=([1], [0]))

        coherences = []
        for first in range(len(amplitudes)):
            for second in range(first + 1, len(amplitudes)):
                left, right = amplitudes[first], amplitudes[second]
                norm = np.linalg.norm(left) * np.linalg.norm(right)
                if norm < 1e-12:
                    continue
                coherences.append(abs(np.vdot(left, right)) / norm)
        if not coherences:
            return 1.0
        return float(np.clip(np.mean(coherences), 0.0, 1.0))

    @staticmethod
    def _validate_profiles(row_profiles: np.ndarray) -> np.ndarray:
        profiles = np.asarray(row_profiles, dtype=np.float32)
        if profiles.ndim != 2:
            raise ValueError("row_profiles must have shape (num_frames, image_height)")
        if not np.isfinite(profiles).all():
            raise ValueError("row_profiles must contain only finite values")
        return profiles

    @staticmethod
    def _empty_metrics() -> RollingBandMetrics:
        return RollingBandMetrics(
            edge_energy=0.0,
            dominant_band_strength=0.0,
            vertical_velocity=0.0,
            position_variance=0.0,
            temporal_periodicity=0.0,
        )
