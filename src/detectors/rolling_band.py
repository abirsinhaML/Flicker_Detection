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
    """Measurements characterizing horizontal bands and their vertical motion."""

    edge_energy: float
    dominant_band_strength: float
    vertical_velocity: float
    position_variance: float
    temporal_periodicity: float


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
        )

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
