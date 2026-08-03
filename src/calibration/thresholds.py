"""Provisional metric normalization used before learned calibration exists.

Detectors measure; this layer decides what a measurement is worth.  Keeping the
two apart is what makes the reference scales below fittable: they are the only
numbers standing between a raw spectral measurement and a severity band, and
they live in configuration rather than in detector bodies.

Every reference is the value at which its component scores 0.5, so a scale can
be read directly as "the measurement I consider borderline".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from src.detectors.awb import AWBMetrics
from src.detectors.illuminant import IlluminantMetrics
from src.detectors.rolling_band import RollingBandMetrics


def soft_saturate(value: float, reference: float) -> float:
    """Map a non-negative measurement onto ``(0, 1)`` via ``v / (v + reference)``.

    Returns exactly 0.5 at ``value == reference`` and rises monotonically without
    ever reaching 1.0.  Unlike a linear ramp with a clamp, this keeps ordering
    intact above the reference: spectral prominence and peak-to-median ratios are
    heavy-tailed, and clamping them collapses every strong artifact onto an
    identical score, erasing the very differences a severity band needs.
    """
    if not isfinite(value) or value <= 0.0:
        return 0.0
    return float(value / (value + reference))


def _mean(components: Sequence[float]) -> float:
    return sum(components) / len(components) if components else 0.0


def _validate_references(**references: float) -> None:
    for name, value in references.items():
        if not isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a finite value greater than zero")


@dataclass(frozen=True, slots=True)
class IlluminantNormalizer:
    """Grade periodic-luminance evidence of illuminant flicker.

    Periodicity and severity are multiplied rather than averaged, because
    flicker requires both and averaging lets either one alone carry a video to a
    middling score.  A clean spectral peak with a 0.1% swing is invisible
    flicker; a large aperiodic swing is camera motion or a scene change.  Only
    the product separates real flicker from both.
    """

    prominence_reference: float
    periodicity_reference: float
    modulation_depth_reference: float

    def __post_init__(self) -> None:
        _validate_references(
            prominence_reference=self.prominence_reference,
            periodicity_reference=self.periodicity_reference,
            modulation_depth_reference=self.modulation_depth_reference,
        )

    def __call__(self, metrics: object) -> float:
        if not isinstance(metrics, IlluminantMetrics):
            raise TypeError("IlluminantNormalizer requires IlluminantMetrics")

        periodic = _mean(
            (
                soft_saturate(metrics.peak_prominence, self.prominence_reference),
                soft_saturate(metrics.peak_to_median_ratio, self.periodicity_reference),
            )
        )
        severity = soft_saturate(metrics.modulation_depth, self.modulation_depth_reference)
        return periodic * severity


@dataclass(frozen=True, slots=True)
class AWBNormalizer:
    """Grade AE/AWB hunting evidence from chroma and low-frequency luma.

    The two channels are combined with ``max`` rather than averaged: AE and AWB
    hunting are alternative mechanisms, and a video exhibiting one strongly
    should not be discounted for lacking the other.
    """

    chroma_prominence_reference: float
    chroma_periodicity_reference: float
    chroma_variance_reference: float
    ae_prominence_reference: float
    ae_periodicity_reference: float
    ae_modulation_depth_reference: float

    def __post_init__(self) -> None:
        _validate_references(
            chroma_prominence_reference=self.chroma_prominence_reference,
            chroma_periodicity_reference=self.chroma_periodicity_reference,
            chroma_variance_reference=self.chroma_variance_reference,
            ae_prominence_reference=self.ae_prominence_reference,
            ae_periodicity_reference=self.ae_periodicity_reference,
            ae_modulation_depth_reference=self.ae_modulation_depth_reference,
        )

    def __call__(self, metrics: object) -> float:
        if not isinstance(metrics, AWBMetrics):
            raise TypeError("AWBNormalizer requires AWBMetrics")

        return max(self._chroma_evidence(metrics), self._ae_evidence(metrics))

    def _chroma_evidence(self, metrics: AWBMetrics) -> float:
        """Periodic colour oscillation, gated by how much colour actually moves.

        The variance gate is a soft multiplier rather than a cutoff, so a nearly
        colourless scene attenuates smoothly instead of dropping to zero at an
        arbitrary boundary.  Decorrelation from luma is what separates AWB
        hunting from illuminant flicker, which moves both channels together.
        """
        periodic = _mean(
            (
                soft_saturate(metrics.chroma_prominence, self.chroma_prominence_reference),
                soft_saturate(metrics.chroma_periodicity, self.chroma_periodicity_reference),
            )
        )
        moving = soft_saturate(metrics.chroma_variance, self.chroma_variance_reference)
        decorrelated = 0.5 + 0.5 * metrics.luma_chroma_decorrelation
        return periodic * moving * decorrelated

    def _ae_evidence(self, metrics: AWBMetrics) -> float:
        """Low-frequency luminance pumping, scored only inside the AE band.

        Gated by modulation depth for the same reason as illuminant flicker: the
        spectral shape measurements are amplitude-blind on their own.
        """
        if not metrics.ae_in_band:
            return 0.0
        periodic = _mean(
            (
                soft_saturate(metrics.ae_prominence, self.ae_prominence_reference),
                soft_saturate(metrics.ae_periodicity, self.ae_periodicity_reference),
            )
        )
        severity = soft_saturate(metrics.ae_modulation_depth, self.ae_modulation_depth_reference)
        return periodic * severity


@dataclass(frozen=True, slots=True)
class RollingBandNormalizer:
    """Map rolling-band measurements to bounded evidence using config scales."""

    band_strength_reference: float
    velocity_reference: float
    position_variance_reference: float
    periodicity_reference: float

    def __post_init__(self) -> None:
        _validate_references(
            band_strength_reference=self.band_strength_reference,
            velocity_reference=self.velocity_reference,
            position_variance_reference=self.position_variance_reference,
            periodicity_reference=self.periodicity_reference,
        )

    def __call__(self, metrics: object) -> float:
        if not isinstance(metrics, RollingBandMetrics):
            raise TypeError("RollingBandNormalizer requires RollingBandMetrics")

        components = (
            metrics.dominant_band_strength / self.band_strength_reference,
            abs(metrics.vertical_velocity) / self.velocity_reference,
            metrics.position_variance / self.position_variance_reference,
            metrics.temporal_periodicity / self.periodicity_reference,
        )
        return _mean([min(max(component, 0.0), 1.0) for component in components])
