"""Chrominance signal extraction."""

from __future__ import annotations

import numpy as np

from src.signals.mask import AnalysisRegion


class ChromaExtractor:
    """Extract temporal chroma signals from Lab frames."""

    @staticmethod
    def lab_signal(
        lab_frames: np.ndarray,
        region: AnalysisRegion,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return live-pixel means of the Lab ``a`` and ``b`` channels per frame.

        Matte borders are colour-neutral, so including them drags chroma
        magnitude toward zero in proportion to the crop and understates AWB
        evidence measured against an absolute reference scale.
        """
        chroma_a = lab_frames[..., 1][:, region.mask].mean(axis=1).astype(np.float32)
        chroma_b = lab_frames[..., 2][:, region.mask].mean(axis=1).astype(np.float32)
        return chroma_a, chroma_b
