"""Detection of periodic illuminant flicker from temporal luminance."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.core.types import FeatureVector
from src.detectors.base import BaseDetector


@dataclass(slots=True)
class IlluminantResult:
    """Frequency-domain evidence of periodic illuminant flicker."""

    score: float
    dominant_frequency: float
    peak_prominence: float
    peak_to_median_ratio: float
    confidence: float


class IlluminantDetector(BaseDetector):
    """Score the strongest non-DC periodic luminance component.

    Candidate frequencies are not constrained to mains frequencies: at ordinary
    video frame rates, the 50/100/120 Hz source signature is commonly observed
    as an alias below the Nyquist limit.
    """

    def __init__(self, min_prominence: float, min_ratio: float) -> None:
        if not isfinite(min_prominence) or min_prominence < 0:
            raise ValueError("min_prominence must be a finite, non-negative value")
        if not isfinite(min_ratio) or min_ratio < 0:
            raise ValueError("min_ratio must be a finite, non-negative value")

        self.min_prominence = min_prominence
        self.min_ratio = min_ratio

    def detect(self, features: FeatureVector) -> IlluminantResult:
        """Evaluate shared frequency metrics without performing DSP."""
        frequency = features.frequency
        score = 0.0
        if frequency.peak_prominence > self.min_prominence:
            score += 0.5
        if frequency.peak_to_median_ratio > self.min_ratio:
            score += 0.5
        score = min(score, 1.0)

        return IlluminantResult(
            score=score,
            dominant_frequency=frequency.dominant_frequency,
            peak_prominence=frequency.peak_prominence,
            peak_to_median_ratio=frequency.peak_to_median_ratio,
            confidence=score,
        )
