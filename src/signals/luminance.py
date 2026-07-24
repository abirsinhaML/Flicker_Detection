"""Luminance signal extraction."""

from __future__ import annotations

import numpy as np


class LuminanceExtractor:
    """Extract temporal luminance signals from YUV frames."""

    @staticmethod
    def global_signal(yuv_frames: np.ndarray) -> np.ndarray:
        """Return one spatially averaged luminance value per frame."""
        return yuv_frames[..., 0].mean(axis=(1, 2)).astype(np.float32)
