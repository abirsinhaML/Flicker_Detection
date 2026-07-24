"""One-pass construction of detector-independent signal features."""

from __future__ import annotations

from src.core.types import SignalFeatures, VideoWindow
from src.signals.chroma import ChromaExtractor
from src.signals.luminance import LuminanceExtractor
from src.signals.preprocessing import SignalPreprocessor
from src.signals.row_profile import RowProfileExtractor


class SignalExtractor:
    """Convert a decoded video window into shared detector inputs."""

    @staticmethod
    def extract(video_window: VideoWindow) -> SignalFeatures:
        """Derive luminance, chroma, and row-profile signals once per window."""
        yuv_frames = SignalPreprocessor.rgb_to_yuv(video_window.frames)
        lab_frames = SignalPreprocessor.rgb_to_lab(video_window.frames)
        chroma_a, chroma_b = ChromaExtractor.lab_signal(lab_frames)

        return SignalFeatures(
            luma=LuminanceExtractor.global_signal(yuv_frames),
            chroma_a=chroma_a,
            chroma_b=chroma_b,
            row_profiles=RowProfileExtractor.extract(yuv_frames),
        )
