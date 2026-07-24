"""Provisional metric normalization used before learned calibration exists."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from src.detectors.rolling_band import RollingBandMetrics


@dataclass(frozen=True, slots=True)
class RollingBandNormalizer:
    """Map rolling-band measurements to bounded evidence using config scales."""

    band_strength_reference: float
    velocity_reference: float
    position_variance_reference: float
    periodicity_reference: float

    def __post_init__(self) -> None:
        for value in (
            self.band_strength_reference,
            self.velocity_reference,
            self.position_variance_reference,
            self.periodicity_reference,
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError("rolling-band normalization references must be positive")

    def __call__(self, metrics: object) -> float:
        if not isinstance(metrics, RollingBandMetrics):
            raise TypeError("RollingBandNormalizer requires RollingBandMetrics")

        components = (
            metrics.dominant_band_strength / self.band_strength_reference,
            abs(metrics.vertical_velocity) / self.velocity_reference,
            metrics.position_variance / self.position_variance_reference,
            metrics.temporal_periodicity / self.periodicity_reference,
        )
        return sum(min(max(component, 0.0), 1.0) for component in components) / len(components)
