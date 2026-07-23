from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


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
class FeatureVector:
    """
    Signals extracted from one VideoWindow.
    """

    luma: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    row_profile: np.ndarray


@dataclass(slots=True)
class VideoWindowResult:
    """
    Flicker assessment for a single temporal VideoWindow.
    """

    start_time: float
    end_time: float

    illuminant_score: float
    awb_score: float
    banding_score: float

    confidence: float


@dataclass(slots=True)
class VideoResult:
    """
    Final assessment for one video.
    """

    video_key: str

    flicker_score: float

    severity: str

    confidence: float

    worst_segment: tuple[float, float]

    detector_version: str