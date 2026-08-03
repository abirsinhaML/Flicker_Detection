from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.features.frequency import FrequencyFeatures
    from src.features.temporal import TemporalSummary


@dataclass(slots=True)
class ManifestEntry:
    """One video to process, identified by a durable source URI.

    ``source_uri`` is either an ``s3://bucket/key`` URI (resolved to a
    short-lived signed URL inside the worker, immediately before decode) or any
    local path / URL FFmpeg accepts directly.  No long-lived signed URL is ever
    stored on disk or carried across processes.
    """

    key: str
    source_uri: str
    size_bytes: int | None = None
    last_modified: datetime | None = None


@dataclass(slots=True)
class VideoMetadata:
    """
    Metadata extracted from a video.
    """

    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


@dataclass(slots=True)
class VideoWindow:
    """
    One sampled temporal VideoWindow from a video.
    """

    start_time: float
    end_time: float
    fps: float
    frames: np.ndarray


@dataclass(slots=True)
class SignalFeatures:
    """Detector-independent signals extracted from one :class:`VideoWindow`.

    ``column_band_profiles`` holds the same row profiles measured over vertical
    slices of the frame, used to test that banding really is horizontal.
    ``valid_fraction`` records how much of the frame carried live pixels.
    """

    luma: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    row_profiles: np.ndarray
    column_band_profiles: np.ndarray
    valid_fraction: float = 1.0


@dataclass(slots=True)
class FeatureVector:
    """Raw signals and shared derived features for one video window."""

    luma: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    row_profiles: np.ndarray
    column_band_profiles: np.ndarray
    fps: float
    temporal: TemporalSummary
    frequency: FrequencyFeatures
    valid_fraction: float = 1.0


@dataclass(slots=True)
class VideoMetrics:
    """Summary of flicker evidence across all sampled video windows."""

    max_score: float
    mean_score: float
    positive_windows: int
    total_windows: int


@dataclass(slots=True)
class VideoResult:
    """Final output for one processed video."""

    video_key: str
    flicker_score: float
    has_flicker: bool
    severity_band: str
    route: str
    confidence: float
    worst_segment: tuple[float, float]
    detector_scores: dict[str, float]
    processing_time: float
    detector_version: str
    # Diagnostics describing measurement conditions rather than flicker evidence.
    horizontal_coherence: float = 1.0
    valid_fraction: float = 1.0
