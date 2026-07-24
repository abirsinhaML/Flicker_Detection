"""Reusable frequency-domain analysis for temporal signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.signal import detrend, find_peaks, welch


@dataclass(slots=True)
class FrequencyFeatures:
    """Welch-spectrum data and its strongest non-DC periodic component."""

    frequencies: np.ndarray
    psd: np.ndarray
    dominant_frequency: float
    dominant_power: float
    peak_prominence: float
    peak_to_median_ratio: float


class FrequencyAnalyzer:
    """Frequency-domain utilities for temporal signals."""

    @staticmethod
    def preprocess(signal: np.ndarray) -> np.ndarray:
        """Linearly detrend and standardize a one-dimensional signal."""
        signal = np.asarray(signal, dtype=np.float32)
        FrequencyAnalyzer._validate_signal(signal)

        if signal.size < 2:
            return np.zeros_like(signal)

        input_scale = max(1.0, float(np.max(np.abs(signal))))
        signal = detrend(signal)
        standard_deviation = signal.std()
        numerical_floor = np.finfo(np.float32).eps * input_scale

        if standard_deviation <= numerical_floor:
            return np.zeros_like(signal)

        return (signal - signal.mean()) / standard_deviation

    @staticmethod
    def power_spectrum(
        signal: np.ndarray,
        fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the Hann-windowed Welch PSD for a temporal signal."""
        FrequencyAnalyzer._validate_fps(fps)
        processed_signal = FrequencyAnalyzer.preprocess(signal)

        if processed_signal.size < 2:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        frequencies, psd = welch(
            processed_signal,
            fs=fps,
            window="hann",
            nperseg=min(256, processed_signal.size),
            scaling="density",
        )
        return frequencies.astype(np.float32), psd.astype(np.float32)

    @staticmethod
    def compute(signal: np.ndarray, fps: float) -> FrequencyFeatures:
        """Compute reusable spectral features from a temporal signal.

        The PSD retains its DC bin for other consumers, while peak selection
        ignores it because brightness level is not periodic flicker evidence.
        """
        frequencies, psd = FrequencyAnalyzer.power_spectrum(signal, fps)
        if frequencies.size < 3:
            return FrequencyAnalyzer._empty_features(frequencies, psd)

        non_dc_psd = psd[frequencies > 0]
        non_dc_frequencies = frequencies[frequencies > 0]
        if non_dc_psd.size < 3 or np.allclose(non_dc_psd, 0.0):
            return FrequencyAnalyzer._empty_features(frequencies, psd)

        peak_indices, properties = find_peaks(non_dc_psd, prominence=0.0)
        if peak_indices.size == 0:
            return FrequencyAnalyzer._empty_features(frequencies, psd)

        strongest_peak = int(np.argmax(properties["prominences"]))
        peak_index = peak_indices[strongest_peak]
        dominant_power = float(non_dc_psd[peak_index])
        peak_prominence = float(properties["prominences"][strongest_peak])
        median_power = float(np.median(non_dc_psd))
        peak_to_median_ratio = dominant_power / max(
            median_power,
            np.finfo(np.float32).eps,
        )

        return FrequencyFeatures(
            frequencies=frequencies,
            psd=psd,
            dominant_frequency=float(non_dc_frequencies[peak_index]),
            dominant_power=dominant_power,
            peak_prominence=peak_prominence,
            peak_to_median_ratio=float(peak_to_median_ratio),
        )

    @staticmethod
    def _validate_signal(signal: np.ndarray) -> None:
        if signal.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if not np.isfinite(signal).all():
            raise ValueError("signal must contain only finite values")

    @staticmethod
    def _validate_fps(fps: float) -> None:
        if not isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite value greater than zero")

    @staticmethod
    def _empty_features(
        frequencies: np.ndarray,
        psd: np.ndarray,
    ) -> FrequencyFeatures:
        return FrequencyFeatures(
            frequencies=frequencies,
            psd=psd,
            dominant_frequency=0.0,
            dominant_power=0.0,
            peak_prominence=0.0,
            peak_to_median_ratio=0.0,
        )
