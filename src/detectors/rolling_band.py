"""Measurement of rolling horizontal bands from luminance row profiles.

A rolling shutter exposes sensor rows in sequence, so an oscillating illuminant
writes its waveform into the frame as a function of row:

    P[t, r] ~ level(r) + A * cos(2*pi * (f*t + k*r + phase))

The temporal frequency ``f`` is the flicker (or its alias); the spatial frequency
``k`` follows from the readout time, ``k = f_true * readout / rows``.  Every
measurement here is taken at ``f``, which is what separates this detector from
the broadband version it replaces.

That earlier version tracked the strongest luminance edge across frames and
measured its energy, velocity, and positional variance.  On egocentric fisheye
footage those quantities are dominated by head motion -- vigorous motion produces
strong row gradients and an edge tracker that jumps between frames -- so it
scored *highest* on clean video and ranked at AUC 0.125 against the reference
labels, inverted.  Projecting onto the flicker frequency first keeps the periodic
component and rejects the broadband motion around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.core.types import FeatureVector
from src.detectors.base import BaseDetector

# Spatial-frequency search grid, in cycles per row, sampled well below one FFT
# bin.  A band spanning less than a single cycle of the frame -- the regime this
# corpus is in -- is sub-bin and invisible to a plain row-wise FFT.
_GRID_OVERSAMPLING = 8
_MIN_CYCLES_PER_FRAME = 0.05


@dataclass(slots=True)
class RollingBandMetrics:
    """Frequency-selective measurements of horizontal banding.

    ``band_amplitude`` is the band's swing as a fraction of scene level, so it is
    comparable across dim and bright footage.  ``phase_linearity`` and
    ``horizontal_coherence`` are already bounded to ``[0, 1]`` and need no
    reference scale: the first asks whether per-row phase advances linearly down
    the frame, the signature of sequential row exposure; the second asks whether
    the frame's vertical slices agree, which holds for a real band and fails when
    the frame has been geometrically remapped.

    ``band_cycles_per_frame`` and ``drift_velocity`` are diagnostics. Zero cycles
    means spatially uniform flicker -- real flicker, but without rolling-shutter
    structure.
    """

    band_amplitude: float
    phase_linearity: float
    band_cycles_per_frame: float
    drift_velocity: float
    horizontal_coherence: float = 1.0


class RollingBandDetector(BaseDetector):
    """Measure banding in a ``(frame, row)`` luminance matrix at the flicker frequency.

    ``drift_velocity`` is signed and expressed in rows per second.  A positive
    value denotes downward motion in image coordinates.
    """

    def __init__(self, smoothing_sigma: float) -> None:
        if not isfinite(smoothing_sigma) or smoothing_sigma <= 0:
            raise ValueError("smoothing_sigma must be a finite value greater than zero")

        self.smoothing_sigma = smoothing_sigma

    def detect(self, features: FeatureVector) -> RollingBandMetrics:
        """Measure band amplitude and spatial phase structure at the flicker frequency."""
        profiles = self._validate_profiles(features.row_profiles)
        flicker_frequency = max(features.frequency.dominant_frequency, 0.0)
        if (
            profiles.shape[0] < 2
            or profiles.shape[1] < 2
            or flicker_frequency <= 0.0
            or not isfinite(features.fps)
            or features.fps <= 0.0
        ):
            # No periodic component means there is no band to characterise.
            return self._empty_metrics()

        # Smoothing along rows suppresses per-row noise.  The bands of interest
        # span a large fraction of the frame, so their phase gradient survives it.
        smoothed = gaussian_filter1d(profiles, sigma=self.smoothing_sigma, axis=1)
        level = float(np.mean(smoothed))
        amplitudes = self._row_amplitudes(smoothed, features.fps, flicker_frequency)
        # Geometry is read before the uniform component is removed and structure
        # after: subtracting the spatial mean biases the estimated band frequency
        # upward for a band spanning under one cycle (a half-cycle band reads as
        # three quarters), while leaving it in lets a row-independent fluctuation
        # masquerade as a band. Measuring each on the array that suits it costs
        # one extra grid search and keeps both honest.
        structured = amplitudes - amplitudes.mean()
        spatial_frequency, _ = self._spatial_phase(amplitudes)
        _, phase_linearity = self._spatial_phase(structured)

        rows = amplitudes.size
        return RollingBandMetrics(
            band_amplitude=self._relative_amplitude(structured, smoothed.shape[0], level),
            phase_linearity=phase_linearity,
            band_cycles_per_frame=float(abs(spatial_frequency) * rows),
            drift_velocity=self._drift_velocity(flicker_frequency, spatial_frequency, rows),
            horizontal_coherence=self._horizontal_coherence(
                features.column_band_profiles,
                features.fps,
                flicker_frequency,
            ),
        )

    @staticmethod
    def _row_amplitudes(
        profiles: np.ndarray,
        fps: float,
        flicker_frequency: float,
    ) -> np.ndarray:
        """Return each row's complex oscillation amplitude at the flicker frequency.

        Phase is retained because it carries the spatial structure: a rolling band
        is precisely a linear phase ramp down the rows.
        """
        frames = profiles.shape[0]
        times = np.arange(frames, dtype=np.float64) / fps
        kernel = np.exp(-2j * np.pi * flicker_frequency * times)
        centered = profiles.astype(np.float64) - profiles.mean(axis=0, keepdims=True)
        return np.asarray(centered.T @ kernel)

    @staticmethod
    def _relative_amplitude(amplitudes: np.ndarray, frames: int, level: float) -> float:
        """Band swing as a fraction of scene level, scale-free across exposures.

        Measured on the spatially structured amplitudes, so a brightness change
        that moves every row in lockstep contributes nothing.  That fluctuation is
        global flicker or exposure pumping and belongs to the illuminant and AWB
        detectors; reading it here is how the previous detector came to score
        motion -- walking under a light produces it readily -- as banding.

        The cost is a known attenuation: a band spanning under a full cycle of the
        frame has a non-zero spatial mean, so some of its own amplitude is removed
        with the uniform component. The bias is consistent across videos and is
        absorbed by the fitted amplitude reference.
        """
        if frames < 1 or level <= np.finfo(np.float32).eps:
            return 0.0
        # A cosine of amplitude A projects onto frames*A/2, so invert that.
        return float(2.0 * np.mean(np.abs(amplitudes)) / (frames * level))

    @staticmethod
    def _spatial_phase(amplitudes: np.ndarray) -> tuple[float, float]:
        """Find the spatial frequency whose phase ramp best explains the rows.

        Returns ``(cycles_per_row, linearity)``.  Linearity is the magnitude of
        the amplitude-weighted phase alignment at the best-fitting spatial
        frequency, so it is 1.0 for a perfectly linear ramp and near 0 for rows
        whose phases are unrelated.  Searching a continuous grid rather than FFT
        bins is what makes a sub-cycle band measurable at all.

        Zero carries no energy by construction, the uniform component having
        already been removed, so no explicit exclusion band is needed.  That
        matters because the bands in this corpus span well under one cycle of the
        frame, and any exclusion wide enough to reject uniform flicker outright
        would have rejected them too.
        """
        magnitude_total = float(np.sum(np.abs(amplitudes)))
        if magnitude_total <= np.finfo(np.float64).eps:
            return 0.0, 0.0

        rows = amplitudes.size
        grid = np.linspace(-0.5, 0.5, _GRID_OVERSAMPLING * rows + 1)
        row_indices = np.arange(rows, dtype=np.float64)
        basis = np.exp(-2j * np.pi * np.outer(grid, row_indices))
        alignment = np.abs(basis @ amplitudes) / magnitude_total
        best = int(np.argmax(alignment))
        return float(grid[best]), float(np.clip(alignment[best], 0.0, 1.0))

    @staticmethod
    def _drift_velocity(
        flicker_frequency: float,
        spatial_frequency: float,
        rows: int,
    ) -> float:
        """Rows per second at which a constant-phase contour travels.

        From ``f*t + k*r = const``, the contour moves at ``-f/k``.  Reported as
        zero for spatially uniform flicker, where the notion does not apply and
        the quotient would diverge.
        """
        if abs(spatial_frequency) * rows < _MIN_CYCLES_PER_FRAME:
            return 0.0
        return float(-flicker_frequency / spatial_frequency)

    @staticmethod
    def _horizontal_coherence(
        column_band_profiles: np.ndarray,
        fps: float,
        flicker_frequency: float,
    ) -> float:
        """Return how consistently vertical slices see the same banding per row.

        Rolling-shutter banding is imposed after the lens has projected the scene,
        so fisheye distortion bends scene content but leaves the bands straight,
        and every slice of the frame should agree on the brightness of each row.

        The comparison is made only at the flicker frequency.  A raw
        frame-to-frame difference is dominated by scene motion, which under a
        fisheye is wildly different at the left and right edges during head
        rotation -- measuring that would report low coherence for every moving
        video regardless of its banding.

        Low coherence alongside real periodic flicker means the frame was
        geometrically remapped -- dewarped, rectilinearized, or electronically
        stabilized -- which bends the bands and invalidates this model.
        """
        profiles = np.asarray(column_band_profiles, dtype=np.float32)
        if profiles.ndim != 3 or profiles.shape[0] < 2 or profiles.shape[1] < 2:
            return 1.0
        if flicker_frequency <= 0.0 or not isfinite(fps) or fps <= 0.0:
            return 1.0

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
            band_amplitude=0.0,
            phase_linearity=0.0,
            band_cycles_per_frame=0.0,
            drift_velocity=0.0,
        )
