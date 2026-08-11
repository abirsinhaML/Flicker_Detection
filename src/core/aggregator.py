"""Calibration-aware aggregation of detector and window-level evidence."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np

from src.core.types import VideoMetrics

ScoreNormalizer = Callable[[object], float]
logger = logging.getLogger(__name__)


class Scored(Protocol):
    """Anything carrying a window score, so the rollup accepts either the bare
    :class:`WindowMetrics` or the fuller :class:`~src.core.types.WindowRecord`
    that is now reported."""

    score: float


@dataclass(slots=True)
class WindowMetrics:
    """Normalized detector evidence and its weighted score for one window."""

    score: float
    detector_scores: dict[str, float]


class DetectionAggregator:
    """Normalize detector metrics, weight them, and summarize windows."""

    def __init__(
        self,
        weights: Mapping[str, float],
        normalizers: Mapping[str, ScoreNormalizer],
        positive_threshold: float,
    ) -> None:
        if not isfinite(positive_threshold) or not 0.0 <= positive_threshold <= 1.0:
            raise ValueError("positive_threshold must be between zero and one")
        if not weights:
            raise ValueError("weights must not be empty")

        self.weights = {name: self._validate_weight(value) for name, value in weights.items()}
        self.normalizers = dict(normalizers)
        self.positive_threshold = positive_threshold

    def aggregate(self, results: Mapping[str, object]) -> WindowMetrics:
        """Normalize and combine the available results for one window."""
        detector_scores: dict[str, float] = {}
        weighted_total = 0.0
        active_weight = 0.0

        for name, result in results.items():
            if name not in self.normalizers:
                raise ValueError(f"No score normalizer configured for {name}")

            score = float(self.normalizers[name](result))
            if not isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Normalizer for {name} returned an invalid score")

            weight = self.weights.get(name, 0.0)
            detector_scores[name] = score
            weighted_total += weight * score
            active_weight += weight

        if active_weight <= 0.0:
            raise ValueError("At least one present detector must have a positive weight")

        metrics = WindowMetrics(
            score=weighted_total / active_weight,
            detector_scores=detector_scores,
        )
        logger.debug("Aggregated window score=%.3f scores=%s", metrics.score, detector_scores)
        return metrics

    def aggregate_video(self, windows: Sequence[Scored]) -> VideoMetrics:
        """Summarize window scores without diluting isolated strong evidence."""
        if not windows:
            raise ValueError("Cannot aggregate a video without sampled windows")

        # float64, so the reported maximum is *exactly* one of the window scores
        # rather than its float32 rounding.  Both numbers are now published side
        # by side, and a consumer checking that the video score is one of its
        # windows would otherwise see a spurious ~1e-8 mismatch.
        scores = np.asarray([window.score for window in windows], dtype=np.float64)
        metrics = VideoMetrics(
            max_score=float(np.max(scores)),
            mean_score=float(np.mean(scores)),
            positive_windows=int(np.count_nonzero(scores >= self.positive_threshold)),
            total_windows=len(windows),
        )
        logger.info(
            "Aggregated %d windows: max_score=%.3f mean_score=%.3f positives=%d",
            metrics.total_windows,
            metrics.max_score,
            metrics.mean_score,
            metrics.positive_windows,
        )
        return metrics

    @staticmethod
    def _validate_weight(weight: float) -> float:
        if not isfinite(weight) or weight < 0.0:
            raise ValueError("detector weights must be finite and non-negative")
        return weight
