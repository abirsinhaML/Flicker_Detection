"""Chrominance signal extraction."""

from __future__ import annotations

import numpy as np


class ChromaExtractor:
    """Extract temporal chroma signals from Lab frames."""

    @staticmethod
    def lab_signal(lab_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return spatial means of the Lab ``a`` and ``b`` channels per frame."""
        chroma_a = lab_frames[..., 1].mean(axis=(1, 2)).astype(np.float32)
        chroma_b = lab_frames[..., 2].mean(axis=(1, 2)).astype(np.float32)
        return chroma_a, chroma_b
