"""Construction of shared derived features from extracted signals."""

from __future__ import annotations

from src.core.types import FeatureVector, SignalFeatures
from src.features.frequency import FrequencyAnalyzer


class FeatureExtractor:
    """Compute reusable temporal and frequency features once per window."""

    @staticmethod
    def extract(
        signals: SignalFeatures,
        fps: float,
        *,
        min_flicker_frequency: float = 0.0,
        max_flicker_frequency: float | None = None,
    ) -> FeatureVector:
        """Return raw signals bundled with their luminance-derived features.

        The flicker band bounds which spectral peak is treated as *the* periodic
        component.  Consumers wanting a different band, such as the AE-hunting
        channel, re-select from the retained PSD rather than re-transforming.
        """
        return FeatureVector(
            luma=signals.luma,
            chroma_a=signals.chroma_a,
            chroma_b=signals.chroma_b,
            row_profiles=signals.row_profiles,
            column_band_profiles=signals.column_band_profiles,
            fps=fps,
            frequency=FrequencyAnalyzer.compute(
                signals.luma,
                fps,
                min_frequency=min_flicker_frequency,
                max_frequency=max_flicker_frequency,
            ),
            valid_fraction=signals.valid_fraction,
        )
