"""Measurement of periodic illuminant flicker from temporal luminance."""

from __future__ import annotations

from dataclasses import dataclass

from src.core.types import FeatureVector
from src.detectors.base import BaseDetector


@dataclass(slots=True)
class IlluminantMetrics:
    """Measurements of the strongest periodic luminance component.

    ``peak_prominence`` and ``peak_to_median_ratio`` say how periodic the
    luminance is; ``modulation_depth`` says how far it actually swings.
    """

    dominant_frequency: float
    dominant_power: float
    peak_prominence: float
    peak_to_median_ratio: float
    modulation_depth: float


class IlluminantDetector(BaseDetector):
    """Report the strongest non-DC periodic luminance component.

    Candidate frequencies are not constrained to mains frequencies: at ordinary
    video frame rates, the 50/100/120 Hz source signature is commonly observed
    as an alias below the Nyquist limit.

    The detector deliberately returns raw measurements and no score.  Deciding
    what a given prominence is *worth* is calibration, and lives in
    :class:`~src.calibration.thresholds.IlluminantNormalizer` so that the scales
    involved can be fitted against reference labels instead of hardcoded here.
    """

    def detect(self, features: FeatureVector) -> IlluminantMetrics:
        """Surface the shared spectral measurements without performing DSP."""
        frequency = features.frequency
        return IlluminantMetrics(
            dominant_frequency=frequency.dominant_frequency,
            dominant_power=frequency.dominant_power,
            peak_prominence=frequency.peak_prominence,
            peak_to_median_ratio=frequency.peak_to_median_ratio,
            modulation_depth=frequency.modulation_depth,
        )
