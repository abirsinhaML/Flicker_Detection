"""Spatial row-profile signal extraction."""

from __future__ import annotations

import numpy as np


class RowProfileExtractor:
    """Extract per-row luminance profiles for rolling-band analysis."""

    @staticmethod
    def extract(yuv_frames: np.ndarray) -> np.ndarray:
        """Return an array shaped ``(num_frames, image_height)``."""
        return yuv_frames[..., 0].mean(axis=2).astype(np.float32)
