"""Common image-space transforms shared by all signal extractors."""

from __future__ import annotations

import cv2
import numpy as np


class SignalPreprocessor:
    """Common preprocessing applied before signal extraction."""

    @staticmethod
    def rgb_to_lab(frames: np.ndarray) -> np.ndarray:
        """Convert an ``(N, H, W, 3)`` RGB frame batch to OpenCV Lab."""
        SignalPreprocessor._validate_frames(frames)
        if not len(frames):
            return frames.copy()

        return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2LAB) for frame in frames])

    @staticmethod
    def rgb_to_yuv(frames: np.ndarray) -> np.ndarray:
        """Convert an ``(N, H, W, 3)`` RGB frame batch to OpenCV YUV."""
        SignalPreprocessor._validate_frames(frames)
        if not len(frames):
            return frames.copy()

        return np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2YUV) for frame in frames])

    @staticmethod
    def gaussian_blur(frames: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """Apply the same odd-sized Gaussian blur to every frame in a batch."""
        SignalPreprocessor._validate_frames(frames)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not len(frames):
            return frames.copy()

        return np.stack(
            [cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0) for frame in frames]
        )

    @staticmethod
    def _validate_frames(frames: np.ndarray) -> None:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (num_frames, height, width, 3)")
