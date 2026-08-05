"""Generic temporal statistics for one-dimensional signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TemporalSummary:
    """Reusable temporal descriptors of a signal."""

    first_difference: np.ndarray
    variance: float
    standard_deviation: float
    rms: float
    energy: float


class TemporalFeatures:
    """Compute generic temporal statistics without detector-specific logic."""

    @staticmethod
    def first_difference(signal: np.ndarray) -> np.ndarray:
        return np.diff(TemporalFeatures._as_signal(signal))

    @staticmethod
    def variance(signal: np.ndarray) -> float:
        signal = TemporalFeatures._as_signal(signal)
        return float(np.var(signal)) if signal.size else 0.0

    @staticmethod
    def std(signal: np.ndarray) -> float:
        signal = TemporalFeatures._as_signal(signal)
        return float(np.std(signal)) if signal.size else 0.0

    @staticmethod
    def rms(signal: np.ndarray) -> float:
        signal = TemporalFeatures._as_signal(signal)
        return float(np.sqrt(np.mean(signal**2))) if signal.size else 0.0

    @staticmethod
    def energy(signal: np.ndarray) -> float:
        signal = TemporalFeatures._as_signal(signal)
        return float(np.sum(signal**2)) if signal.size else 0.0

    @staticmethod
    def summarize(signal: np.ndarray) -> TemporalSummary:
        """Bundle common temporal descriptors for a signal."""
        signal = TemporalFeatures._as_signal(signal)
        if not signal.size:
            return TemporalSummary(
                first_difference=np.empty(0, dtype=np.float32),
                variance=0.0,
                standard_deviation=0.0,
                rms=0.0,
                energy=0.0,
            )

        return TemporalSummary(
            first_difference=np.diff(signal),
            variance=float(np.var(signal)),
            standard_deviation=float(np.std(signal)),
            rms=float(np.sqrt(np.mean(signal**2))),
            energy=float(np.sum(signal**2)),
        )

    @staticmethod
    def _as_signal(signal: np.ndarray) -> np.ndarray:
        values = np.asarray(signal, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("signal must contain only finite values")
        return values
