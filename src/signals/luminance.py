"""Luminance signal extraction."""

from __future__ import annotations

import numpy as np

from src.signals.mask import AnalysisRegion


class LuminanceExtractor:
    """Extract temporal luminance signals from YUV frames."""

    @staticmethod
    def global_signal(yuv_frames: np.ndarray, region: AnalysisRegion) -> np.ndarray:
        """Return one luminance value per frame, averaged over live pixels only.

        Dead border pixels scale both the mean and its fluctuation by the same
        factor, so they cancel out of modulation depth.  They do not cancel out
        of absolute measurements, which is why they are excluded here rather
        than corrected for downstream.
        """
        return yuv_frames[..., 0][:, region.mask].mean(axis=1).astype(np.float32)
