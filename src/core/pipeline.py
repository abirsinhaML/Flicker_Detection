"""Detector-agnostic orchestration for one precomputed feature vector."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from src.core.types import FeatureVector
from src.detectors.base import BaseDetector

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Run a configured collection of detectors against shared features."""

    def __init__(self, detectors: Sequence[BaseDetector]) -> None:
        self.detectors = list(detectors)

    def run(self, features: FeatureVector) -> dict[str, object]:
        """Return each detector's result keyed by its class name."""
        results: dict[str, object] = {}
        for detector in self.detectors:
            name = type(detector).__name__
            results[name] = detector.detect(features)
            logger.debug("Completed %s", name)
        return results
