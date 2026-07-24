"""Construction of shared derived features from extracted signals."""

from __future__ import annotations

from src.core.types import FeatureVector, SignalFeatures
from src.features.frequency import FrequencyAnalyzer
from src.features.temporal import TemporalFeatures


class FeatureExtractor:
    """Compute reusable temporal and frequency features once per window."""

    @staticmethod
    def extract(signals: SignalFeatures, fps: float) -> FeatureVector:
        """Return raw signals bundled with their luminance-derived features."""
        return FeatureVector(
            luma=signals.luma,
            chroma_a=signals.chroma_a,
            chroma_b=signals.chroma_b,
            row_profiles=signals.row_profiles,
            fps=fps,
            temporal=TemporalFeatures.summarize(signals.luma),
            frequency=FrequencyAnalyzer.compute(signals.luma, fps),
        )
