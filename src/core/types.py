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
    """
    Represents one row in the dataset manifest.
    """

    key: str
    size_bytes: int
    last_modified: datetime
    presigned_url: str


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
    """Detector-independent signals extracted from one :class:`VideoWindow`."""

    luma: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    row_profiles: np.ndarray


@dataclass(slots=True)
class FeatureVector:
    """Raw signals and shared derived features for one video window."""

    luma: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    row_profiles: np.ndarray
    fps: float
    temporal: TemporalSummary
    frequency: FrequencyFeatures



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
