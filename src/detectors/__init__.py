"""Flicker detectors built on shared signal features.

Every detector returns raw measurements.  Converting a measurement into bounded
evidence is calibration, and lives in :mod:`src.calibration.thresholds`.
"""

from src.detectors.awb import AWBDetector, AWBMetrics
from src.detectors.base import BaseDetector
from src.detectors.illuminant import IlluminantDetector, IlluminantMetrics
from src.detectors.rolling_band import RollingBandDetector, RollingBandMetrics

__all__ = [
    "AWBDetector",
    "AWBMetrics",
    "BaseDetector",
    "IlluminantDetector",
    "IlluminantMetrics",
    "RollingBandDetector",
    "RollingBandMetrics",
]
