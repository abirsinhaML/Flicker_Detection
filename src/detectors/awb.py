"""Auto-exposure and auto-white-balance hunting detector.

AE hunting manifests as low-frequency brightness pumping driven by the camera's
auto-exposure controller oscillating around a set-point.  AWB hunting produces
periodic color shifts as the white-balance algorithm hunts, typically
uncorrelated with the luminance signal.

Both are distinguished from illuminant flicker by their lower temporal
frequency and, for AWB, by chroma oscillation that is decorrelated from luma.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from src.core.types import FeatureVector
from src.detectors.base import BaseDetector
from src.features.frequency import FrequencyAnalyzer


@dataclass(slots=True)
class AWBResult:
    """Evidence of auto-exposure or auto-white-balance hunting."""

    score: float
    chroma_periodicity: float
    chroma_variance: float
    ae_evidence: float
    luma_chroma_decorrelation: float


class AWBDetector(BaseDetector):
    """Detect AE/AWB hunting from chroma and low-frequency luma oscillation.

    The detector evaluates two independent evidence channels:

    1. **Chroma hunting** — periodic oscillation in Lab a*/b* magnitude that is
       decorrelated from luminance.  True illuminant flicker modulates both
       luma and chroma together, whereas AWB hunting shifts colour independently.

    2. **AE hunting** — low-frequency (below ``ae_max_frequency``) periodic
       luminance oscillation that falls outside the mains-beat band targeted
       by :class:`IlluminantDetector`.

    The final score is the maximum of the two channels, clamped to [0, 1].
    """

    def __init__(
        self,
        min_chroma_std: float = 0.5,
        ae_max_frequency: float = 5.0,
    ) -> None:
        if not isfinite(min_chroma_std) or min_chroma_std < 0:
            raise ValueError("min_chroma_std must be a finite, non-negative value")
        if not isfinite(ae_max_frequency) or ae_max_frequency <= 0:
            raise ValueError("ae_max_frequency must be a finite value greater than zero")

        self.min_chroma_std = min_chroma_std
        self.ae_max_frequency = ae_max_frequency

    def detect(self, features: FeatureVector) -> AWBResult:
        """Score AE/AWB evidence from shared feature signals."""
        chroma_score, chroma_periodicity, chroma_var = self._chroma_hunting(features)
        ae_score = self._ae_hunting(features)
        decorrelation = self._luma_chroma_decorrelation(features)

        # Boost chroma score when decorrelated from luma (true AWB, not illuminant)
        adjusted_chroma = chroma_score * (0.5 + 0.5 * decorrelation)

        combined = float(np.clip(max(adjusted_chroma, ae_score), 0.0, 1.0))

        return AWBResult(
            score=combined,
            chroma_periodicity=chroma_periodicity,
            chroma_variance=chroma_var,
            ae_evidence=ae_score,
            luma_chroma_decorrelation=decorrelation,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chroma_hunting(self, features: FeatureVector) -> tuple[float, float, float]:
        """Score periodic oscillation in the chroma channels."""
        chroma_a = np.asarray(features.chroma_a, dtype=np.float32)
        chroma_b = np.asarray(features.chroma_b, dtype=np.float32)

        if chroma_a.size < 8:
            return 0.0, 0.0, 0.0

        chroma_mag = np.sqrt(chroma_a**2 + chroma_b**2)
        chroma_std = float(np.std(chroma_mag))

        if chroma_std < self.min_chroma_std:
            return 0.0, 0.0, chroma_std

        freq_features = FrequencyAnalyzer.compute(chroma_mag, features.fps)
        periodicity = freq_features.peak_to_median_ratio
        prominence = freq_features.peak_prominence

        # Score: strong periodic component in chroma → AWB hunting evidence
        score = 0.0
        if prominence > 2.0:
            score += 0.5
        if periodicity > 4.0:
            score += 0.5

        return min(score, 1.0), periodicity, chroma_std

    def _ae_hunting(self, features: FeatureVector) -> float:
        """Score low-frequency periodic luma oscillation (AE hunting).

        Complements the illuminant detector by targeting slow oscillation
        below the expected mains-beat frequencies.  If the dominant luminance
        frequency is above ``ae_max_frequency`` it is left to the illuminant
        detector.
        """
        freq = features.frequency

        # Only flag if the dominant frequency is low (AE hunting range)
        if freq.dominant_frequency <= 0 or freq.dominant_frequency > self.ae_max_frequency:
            return 0.0

        # Require meaningful prominence and periodicity
        score = 0.0
        if freq.peak_prominence > 2.0:
            score += 0.5
        if freq.peak_to_median_ratio > 4.0:
            score += 0.5

        return min(score, 1.0)

    @staticmethod
    def _luma_chroma_decorrelation(features: FeatureVector) -> float:
        """Measure how independent chroma oscillation is from luma.

        Returns a value between 0.0 (perfectly correlated) and 1.0
        (completely independent).  True AWB hunting typically shows chroma
        oscillation that is decorrelated from luma, while illuminant flicker
        modulates both channels together.
        """
        luma = np.asarray(features.luma, dtype=np.float32)
        chroma_a = np.asarray(features.chroma_a, dtype=np.float32)
        chroma_b = np.asarray(features.chroma_b, dtype=np.float32)

        if luma.size < 4 or luma.size != chroma_a.size:
            return 0.5  # insufficient data → neutral prior

        chroma_mag = np.sqrt(chroma_a**2 + chroma_b**2)

        luma_std = float(np.std(luma))
        chroma_std = float(np.std(chroma_mag))

        if luma_std < 1e-6 or chroma_std < 1e-6:
            return 0.5  # flat signal → neutral prior

        corr = float(np.corrcoef(luma, chroma_mag)[0, 1])
        if not np.isfinite(corr):
            return 0.5

        return 1.0 - abs(corr)
