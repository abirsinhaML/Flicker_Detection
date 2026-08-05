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

# Below this many frames a windowed spectrum is not meaningful.
_MIN_CHROMA_FRAMES = 8
# Correlation is undefined for a flat signal; treat it as no evidence either way.
_NEUTRAL_DECORRELATION = 0.5


@dataclass(slots=True)
class AWBMetrics:
    """Measurements of auto-exposure and auto-white-balance hunting.

    Two independent evidence channels, both reported raw:

    * chroma — periodic oscillation in Lab a*/b* magnitude and how strongly it
      moves, plus its independence from luma
    * AE — periodic luminance oscillation, and whether the dominant luminance
      frequency falls inside the auto-exposure band at all
    """

    chroma_prominence: float
    chroma_periodicity: float
    chroma_variance: float
    luma_chroma_decorrelation: float
    ae_prominence: float
    ae_periodicity: float
    ae_modulation_depth: float
    ae_dominant_frequency: float
    ae_in_band: bool


class AWBDetector(BaseDetector):
    """Measure AE/AWB hunting from chroma and low-frequency luma oscillation.

    The AE band separates two *mechanisms* rather than grading severity:
    auto-exposure controllers hunt at a few hertz, while the mains beat sits
    higher and belongs to :class:`~src.detectors.illuminant.IlluminantDetector`.
    The band therefore stays here as a definition, while the magnitude of the
    evidence is graded in
    :class:`~src.calibration.thresholds.AWBNormalizer`.

    ``ae_min_frequency`` excludes the slowest content, which is dominated by
    egocentric head motion and exposure drift rather than controller hunting.
    The separation is imperfect: genuine AE hunting and head motion overlap in
    this band, and nothing here can fully distinguish them.
    """

    def __init__(self, ae_min_frequency: float = 0.5, ae_max_frequency: float = 5.0) -> None:
        if not isfinite(ae_max_frequency) or ae_max_frequency <= 0:
            raise ValueError("ae_max_frequency must be a finite value greater than zero")
        if not isfinite(ae_min_frequency) or ae_min_frequency < 0:
            raise ValueError("ae_min_frequency must be a finite, non-negative value")
        if ae_min_frequency >= ae_max_frequency:
            raise ValueError("ae_min_frequency must be below ae_max_frequency")

        self.ae_min_frequency = ae_min_frequency
        self.ae_max_frequency = ae_max_frequency

    def detect(self, features: FeatureVector) -> AWBMetrics:
        """Measure both hunting channels without scoring either."""
        chroma_prominence, chroma_periodicity, chroma_variance = self._chroma_oscillation(features)
        # The AE channel asks the retained spectrum for its own band rather than
        # reusing the flicker peak, which is deliberately selected above it.
        ae_peak = features.frequency.peak_in_band(
            min_frequency=self.ae_min_frequency,
            max_frequency=self.ae_max_frequency,
        )

        return AWBMetrics(
            chroma_prominence=chroma_prominence,
            chroma_periodicity=chroma_periodicity,
            chroma_variance=chroma_variance,
            luma_chroma_decorrelation=self._luma_chroma_decorrelation(features),
            ae_prominence=ae_peak.prominence,
            ae_periodicity=ae_peak.peak_to_median_ratio,
            ae_modulation_depth=features.frequency.modulation_depth,
            ae_dominant_frequency=ae_peak.frequency,
            ae_in_band=ae_peak.found,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chroma_oscillation(features: FeatureVector) -> tuple[float, float, float]:
        """Measure periodic oscillation in the chroma magnitude signal."""
        chroma_a = np.asarray(features.chroma_a, dtype=np.float32)
        chroma_b = np.asarray(features.chroma_b, dtype=np.float32)

        if chroma_a.size < _MIN_CHROMA_FRAMES:
            return 0.0, 0.0, 0.0

        chroma_magnitude = np.sqrt(chroma_a**2 + chroma_b**2)
        chroma_variance = float(np.std(chroma_magnitude))
        spectrum = FrequencyAnalyzer.compute(chroma_magnitude, features.fps)
        return (
            spectrum.peak_prominence,
            spectrum.peak_to_median_ratio,
            chroma_variance,
        )

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
            return _NEUTRAL_DECORRELATION

        chroma_magnitude = np.sqrt(chroma_a**2 + chroma_b**2)
        if float(np.std(luma)) < 1e-6 or float(np.std(chroma_magnitude)) < 1e-6:
            return _NEUTRAL_DECORRELATION

        correlation = float(np.corrcoef(luma, chroma_magnitude)[0, 1])
        if not np.isfinite(correlation):
            return _NEUTRAL_DECORRELATION

        return 1.0 - abs(correlation)
