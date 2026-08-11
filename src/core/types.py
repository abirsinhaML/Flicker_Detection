from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.features.frequency import FrequencyFeatures


@dataclass(slots=True)
class ManifestEntry:
    """One video to process, identified by a durable source URI.

    ``source_uri`` is either an ``s3://bucket/key`` URI (resolved to a
    short-lived signed URL inside the worker, immediately before decode) or any
    local path / URL FFmpeg accepts directly.  No long-lived signed URL is ever
    stored on disk or carried across processes.

    ``project_name``, ``video_id``, and ``duration_seconds`` are carried from the
    link sheet when it supplies them.  They travel with the entry so the output
    can be grouped by project without joining back to the spreadsheet, and so the
    sheet's stated duration can be checked against the one the decoder reports.
    ``video_id`` is not unique in this corpus and is never used to identify a
    video; the S3 key is.
    """

    key: str
    source_uri: str
    size_bytes: int | None = None
    last_modified: datetime | None = None
    project_name: str | None = None
    video_id: str | None = None
    duration_seconds: float | None = None


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
class WindowRecord:
    """Everything measured and decided about one sampled window.

    This is the pipeline's reporting unit.  A video-level row can only carry the
    worst window's evidence, which discards the other 199: how many windows were
    affected, whether the artifact was continuous or a single moment, and where in
    the timeline it sat.  All of that is answerable per window and unanswerable
    from a maximum.

    ``detector_scores`` is keyed by detector class name, as the aggregator's
    weights are, so the weighted sum of the entries reproduces ``score``
    exactly.  ``measurements`` holds each detector's raw, pre-normalization
    output keyed by detector short name, so a score can always be traced back to
    the quantity that produced it without re-running the decode.

    ``severity_band`` / ``route`` / ``confidence`` here grade *this window* under
    the window thresholds, which are a separate calibration from the video ones:
    a video score is a maximum over ~200 windows and therefore sits far up the
    per-window distribution.
    """

    index: int
    start_time: float
    end_time: float
    fps: float
    frame_count: int
    score: float
    severity_band: str
    route: str
    confidence: float
    detector_scores: dict[str, float]
    measurements: dict[str, dict[str, float | bool]]
    # Diagnostics describing measurement conditions rather than flicker evidence.
    valid_fraction: float = 1.0
    horizontal_coherence: float = 1.0
    is_worst: bool = False


@dataclass(slots=True)
class VideoResult:
    """Final output for one processed video: its rollup and every window."""

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
    # Per-window evidence, in time order.  The rollup fields above are all
    # derived from these, so a consumer never has to trust a summary it cannot
    # recompute.
    windows: list[WindowRecord] = field(default_factory=list)
    mean_score: float = 0.0
    positive_windows: int = 0
    total_windows: int = 0
    worst_index: int = 0
    duration: float = 0.0
    video_fps: float = 0.0
    width: int = 0
    height: int = 0
    decode_backend: str = ""
    # Carried from the link sheet, not measured here.  ``sheet_duration`` is what
    # the sheet claims; ``duration`` is what the container reports, and the two
    # disagreeing is worth seeing rather than reconciling silently.
    project_name: str = ""
    video_id: str = ""
    sheet_duration: float | None = None
