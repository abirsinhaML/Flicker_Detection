"""One-pass construction of detector-independent signal features."""

from __future__ import annotations

from src.core.types import SignalFeatures, VideoWindow
from src.signals.chroma import ChromaExtractor
from src.signals.luminance import LuminanceExtractor
from src.signals.mask import AnalysisRegion, RegionPolicy
from src.signals.preprocessing import SignalPreprocessor
from src.signals.row_profile import RowProfileExtractor


class SignalExtractor:
    """Convert a decoded video window into shared detector inputs."""

    @staticmethod
    def extract(
        video_window: VideoWindow,
        policy: RegionPolicy | None = None,
    ) -> SignalFeatures:
        """Derive luminance, chroma, and row-profile signals once per window.

        Every signal is measured over the same region, so they stay mutually
        comparable and none of them inherits the frame's dead borders.
        """
        yuv_frames = SignalPreprocessor.rgb_to_yuv(video_window.frames)
        lab_frames = SignalPreprocessor.rgb_to_lab(video_window.frames)
        region = AnalysisRegion.detect(yuv_frames[..., 0], policy or RegionPolicy())
        chroma_a, chroma_b = ChromaExtractor.lab_signal(lab_frames, region)

        return SignalFeatures(
            luma=LuminanceExtractor.global_signal(yuv_frames, region),
            chroma_a=chroma_a,
            chroma_b=chroma_b,
            row_profiles=RowProfileExtractor.extract(yuv_frames, region),
            column_band_profiles=RowProfileExtractor.extract_column_bands(yuv_frames, region),
            valid_fraction=region.valid_fraction,
        )
