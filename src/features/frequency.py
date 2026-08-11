"""Reusable frequency-domain analysis for temporal signals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.signal import detrend, find_peaks, welch


@dataclass(slots=True)
class BandPeak:
    """The strongest periodic component inside one frequency band."""

    frequency: float
    power: float
    prominence: float
    peak_to_median_ratio: float

    @classmethod
    def empty(cls) -> BandPeak:
        return cls(frequency=0.0, power=0.0, prominence=0.0, peak_to_median_ratio=0.0)

    @property
    def found(self) -> bool:
        return self.frequency > 0.0


@dataclass(slots=True)
class FrequencyFeatures:
    """Welch-spectrum data and its strongest non-DC periodic component.

    ``peak_prominence`` and ``peak_to_median_ratio`` describe the *shape* of the
    spectrum, measured after standardization, so they answer "is this periodic?"
    and are deliberately blind to amplitude.  ``modulation_depth`` carries the
    amplitude information they discard and answers "how severe is it?".  Flicker
    severity needs both: a perfectly periodic 0.1% ripple is invisible, and a
    large aperiodic swing is camera motion.
    """

    frequencies: np.ndarray
    psd: np.ndarray
    dominant_frequency: float
    dominant_power: float
    peak_prominence: float
    peak_to_median_ratio: float
    modulation_depth: float

    def peak_in_band(
        self,
        min_frequency: float = 0.0,
        max_frequency: float | None = None,
    ) -> BandPeak:
        """Return the strongest peak within a band, reusing this spectrum.

        Lets a second consumer ask a different question of the same PSD without
        another Welch transform -- the AE-hunting channel wants the low-frequency
        peak that the flicker band deliberately excludes.
        """
        return FrequencyAnalyzer.select_peak(
            self.frequencies,
            self.psd,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )


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
    def compute(
        signal: np.ndarray,
        fps: float,
        *,
        min_frequency: float = 0.0,
        max_frequency: float | None = None,
    ) -> FrequencyFeatures:
        """Compute reusable spectral features from a temporal signal.

        The PSD retains its DC bin for other consumers, while peak selection
        ignores it because brightness level is not periodic flicker evidence.

        ``min_frequency`` bounds peak selection from below, which matters more
        than it sounds.  Over a window long enough to resolve the flicker band,
        egocentric head motion and exposure drift deposit far more energy below
        2 Hz than the flicker itself carries, so an unbounded search reports the
        motion as the dominant component and every frequency-domain measurement
        then describes the wrong phenomenon.
        """
        modulation_depth = FrequencyAnalyzer.modulation_depth(signal)
        frequencies, psd = FrequencyAnalyzer.power_spectrum(signal, fps)
        peak = FrequencyAnalyzer.select_peak(
            frequencies,
            psd,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
        )

        return FrequencyFeatures(
            frequencies=frequencies,
            psd=psd,
            dominant_frequency=peak.frequency,
            dominant_power=peak.power,
            peak_prominence=peak.prominence,
            peak_to_median_ratio=peak.peak_to_median_ratio,
            modulation_depth=modulation_depth,
        )

    @staticmethod
    def select_peak(
        frequencies: np.ndarray,
        psd: np.ndarray,
        *,
        min_frequency: float = 0.0,
        max_frequency: float | None = None,
    ) -> BandPeak:
        """Return the most prominent spectral peak inside a frequency band.

        Peaks are found across the whole non-DC spectrum and only then filtered
        to the band, so prominence is measured against a peak's real neighbourhood
        rather than against a window edge that the band happens to cut.  The
        noise floor is likewise the median of the whole non-DC spectrum: a median
        taken inside a narrow band can be dominated by the peak being measured.
        """
        if frequencies.size < 3 or psd.size != frequencies.size:
            return BandPeak.empty()

        non_dc = frequencies > 0
        non_dc_psd = psd[non_dc]
        non_dc_frequencies = frequencies[non_dc]
        if non_dc_psd.size < 3 or np.allclose(non_dc_psd, 0.0):
            return BandPeak.empty()

        peak_indices, properties = find_peaks(non_dc_psd, prominence=0.0)
        if peak_indices.size == 0:
            return BandPeak.empty()

        peak_frequencies = non_dc_frequencies[peak_indices]
        in_band = peak_frequencies >= min_frequency
        if max_frequency is not None:
            in_band &= peak_frequencies <= max_frequency
        if not in_band.any():
            return BandPeak.empty()

        prominences = properties["prominences"]
        candidates = np.flatnonzero(in_band)
        strongest = candidates[int(np.argmax(prominences[candidates]))]
        peak_index = peak_indices[strongest]
        power = float(non_dc_psd[peak_index])
        median_power = float(np.median(non_dc_psd))

        return BandPeak(
            frequency=float(non_dc_frequencies[peak_index]),
            power=power,
            prominence=float(prominences[strongest]),
            peak_to_median_ratio=float(power / max(median_power, np.finfo(np.float32).eps)),
        )

    @staticmethod
    def modulation_depth(signal: np.ndarray) -> float:
        """Return the signal's relative fluctuation about its own level.

        This is the coefficient of variation of the linearly detrended signal:
        the physically meaningful severity axis for flicker, and scale-free, so
        a dim scene and a bright one are comparable.  Detrending first means a
        slow exposure ramp does not register as flicker.
        """
        signal = np.asarray(signal, dtype=np.float32)
        FrequencyAnalyzer._validate_signal(signal)
        if signal.size < 2:
            return 0.0

        level = float(np.mean(np.abs(signal)))
        if level <= np.finfo(np.float32).eps:
            return 0.0
        return float(np.std(detrend(signal)) / level)

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
