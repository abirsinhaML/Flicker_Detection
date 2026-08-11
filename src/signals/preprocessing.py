"""Common image-space transforms shared by all signal extractors."""

from __future__ import annotations

import cv2
import numpy as np


class SignalPreprocessor:
    """Common preprocessing applied before signal extraction."""

    @staticmethod
    def rgb_to_lab(frames: np.ndarray) -> np.ndarray:
        """Convert an ``(N, H, W, 3)`` RGB frame batch to OpenCV Lab."""
        return SignalPreprocessor._convert(frames, cv2.COLOR_RGB2LAB)

    @staticmethod
    def rgb_to_yuv(frames: np.ndarray) -> np.ndarray:
        """Convert an ``(N, H, W, 3)`` RGB frame batch to OpenCV YUV."""
        return SignalPreprocessor._convert(frames, cv2.COLOR_RGB2YUV)

    @staticmethod
    def _convert(frames: np.ndarray, code: int) -> np.ndarray:
        """Colour-convert a whole batch in one OpenCV call.

        These conversions are per-pixel with no spatial neighbourhood, so a
        batch stacked into a single tall image gives bit-identical output to
        converting each frame separately -- while spending one call into
        OpenCV's threaded, vectorised path instead of ``N`` of them.  At the
        analysed window size that is 12.2 ms -> 1.4 ms for YUV and 58.0 ms ->
        21.5 ms for Lab, the largest remaining CPU cost after decoding.
        """
        SignalPreprocessor._validate_frames(frames)
        if not len(frames):
            return frames.copy()

        count, height, width, channels = frames.shape
        stacked = np.ascontiguousarray(frames).reshape(count * height, width, channels)
        return cv2.cvtColor(stacked, code).reshape(count, height, width, channels)

    @staticmethod
    def _validate_frames(frames: np.ndarray) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (num_frames, height, width, 3)")
