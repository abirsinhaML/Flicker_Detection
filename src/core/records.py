"""Serialization of window and video evidence into stable output records.

The pipeline reports per window.  A video-level maximum answers "is anything
wrong with this video" and nothing else: it cannot say how many windows were
affected, whether the artifact was continuous or a single moment, or where in
the timeline it sat.  Those are the questions a reviewer actually asks, and each
of them is a per-window question.

Two shapes are produced from the same records:

* a nested JSON object per video, which is the durable output.  One video is one
  line, so a batch stays append-only and resumable at video granularity even
  though it now writes tens of rows per video.
* flat rows, for loading into a dataframe.  The window table is the analytics
  unit; the video table is the rollup, kept in the original flag-manifest schema
  so the existing calibration and progress tooling reads it unchanged.

Measurement columns are derived from the detector metric dataclasses themselves
rather than restated here, so a detector that gains a measurement cannot quietly
stop reporting it.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from src.core.types import VideoResult, WindowRecord
from src.detectors.awb import AWBMetrics
from src.detectors.illuminant import IlluminantMetrics
from src.detectors.rolling_band import RollingBandMetrics

# Bumped from 1.2: the output is now one record per video containing one entry
# per window, rather than a single flat row.
SCHEMA_VERSION = "2.0"

# Detectors whose measurements appear as columns in the flat window table.  The
# nested JSON carries whatever detectors actually ran; a fixed column list keeps
# the CSV schema stable, which is what a downstream consumer needs.
_TABULATED_METRICS = (
    ("illuminant", IlluminantMetrics),
    ("rolling_band", RollingBandMetrics),
    ("awb", AWBMetrics),
)

# The rollup schema, unchanged from the per-video CSV so that
# src/calibration/*.py and scripts/check_prefix_progress.py keep working, plus
# the window counts that only exist now that windows are retained.
VIDEO_FIELDS: tuple[str, ...] = (
    "schema_version",
    "status",
    "error",
    "video_key",
    "flicker_score",
    "severity_band",
    "route",
    "confidence",
    "worst_segment_start",
    "worst_segment_end",
    "illuminant_score",
    "rolling_band_score",
    "awb_score",
    # Diagnostics, not evidence: they qualify how far the scores can be trusted.
    "horizontal_coherence",
    "valid_fraction",
    "processing_time_seconds",
    "detector_version",
    # Appended, so an older manifest still parses and resumes; rows written
    # before this column existed simply carry an empty value.
    "completed_at",
    # Appended for the same reason: window-level context for the maximum above.
    "total_windows",
    "positive_windows",
    "mean_score",
    "worst_window_index",
    "windows_mild",
    "windows_extreme",
    "duration_seconds",
    # Carried from the link sheet, so results group by project without a join
    # back to the spreadsheet.
    "project_name",
    "video_id",
)


def _detector_key(detector_name: str) -> str:
    """Map a detector class name to its output key.

    ``RollingBandDetector`` becomes ``rolling_band`` and ``AWBDetector`` becomes
    ``awb``.  Deriving the key means adding a detector needs no change here.
    """
    stem = re.sub(r"Detector$", "", detector_name)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", stem)
    return spaced.lower()


def _measurement_columns() -> tuple[str, ...]:
    """Flat column names for every tabulated detector measurement."""
    return tuple(
        f"{key}_{field.name}"
        for key, metrics_type in _TABULATED_METRICS
        for field in fields(metrics_type)
    )


WINDOW_FIELDS: tuple[str, ...] = (
    "schema_version",
    "video_key",
    "window_index",
    "start_time",
    "end_time",
    "fps",
    "frame_count",
    "window_score",
    "severity_band",
    "route",
    "confidence",
    "is_worst",
    "illuminant_score",
    "rolling_band_score",
    "awb_score",
    "valid_fraction",
    "horizontal_coherence",
    *_measurement_columns(),
    # Video context, repeated on every window row so the table stands alone.
    "project_name",
    "video_flicker_score",
    "video_severity_band",
    "video_route",
    "total_windows",
    "detector_version",
    "completed_at",
)


def window_csv_path(video_key: str, root: str | Path) -> Path:
    """Map a video key to its own window-metrics CSV beneath ``root``.

    The key's directories are reproduced verbatim and only the extension
    changes, so a result sits at the same address as the video it describes::

        raw/Delhi_ZetWork/2026-06-18/WRK-73637/GX060044.MP4
        -> <root>/raw/Delhi_ZetWork/2026-06-18/WRK-73637/GX060044.csv

    Keys come from a bucket listing rather than from this process, so they are
    treated as untrusted input: a leading slash or a ``..`` segment would
    otherwise place the file outside ``root``, silently overwriting something
    else on the machine.  Both are rejected rather than sanitised, because a key
    that needs rewriting to be safe is one whose output path no longer matches
    the video it came from.
    """
    key = str(video_key).strip().replace("\\", "/").lstrip("/")
    parts = [part for part in key.split("/") if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe video key for an output path: {video_key!r}")
    # Only the final extension is replaced, and the name is rebuilt rather than
    # passed through ``with_suffix``: that strips whatever follows the last dot,
    # so ``clip.v2.MP4`` and ``clip.MP4`` would both land on ``clip.csv`` and one
    # would silently overwrite the other.
    stem = PurePosixPath(parts[-1]).stem or parts[-1]
    parts[-1] = f"{stem}.csv"
    return Path(root).joinpath(*parts)


def scalar(value: Any) -> Any:
    """Convert a NumPy scalar to the plain Python type JSON and CSV expect."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def measurements_of(results: Mapping[str, object]) -> dict[str, dict[str, float | bool]]:
    """Extract every detector's raw measurements, keyed by detector short name.

    Read generically off the metric dataclasses, so a detector's new measurement
    is reported without a change here.  Array-valued fields are skipped: the
    retained PSD is a per-window array and belongs in a diagnostic dump, not in
    a record written once per window for a quarter-million videos.
    """
    extracted: dict[str, dict[str, float | bool]] = {}
    for name, metrics in results.items():
        if not is_dataclass(metrics) or isinstance(metrics, type):
            continue
        extracted[_detector_key(name)] = {
            field.name: scalar(getattr(metrics, field.name))
            for field in fields(metrics)
            if not isinstance(getattr(metrics, field.name), np.ndarray)
        }
    return extracted


def result_to_record(result: VideoResult, *, completed_at: str) -> dict[str, Any]:
    """Build the durable nested record for one successfully processed video."""
    scores = {_detector_key(name): scalar(score) for name, score in result.detector_scores.items()}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "error": "",
        "video_key": result.video_key,
        "detector_version": result.detector_version,
        "completed_at": completed_at,
        "processing_time_seconds": scalar(result.processing_time),
        "source": {
            "duration": scalar(result.duration),
            "fps": scalar(result.video_fps),
            "width": int(result.width),
            "height": int(result.height),
            "decode_backend": result.decode_backend,
            # From the link sheet rather than the container.  Kept beside
            # ``duration`` instead of replacing it, so a sheet that disagrees
            # with the file is visible instead of reconciled away.
            "project_name": result.project_name,
            "video_id": result.video_id,
            "sheet_duration": scalar(result.sheet_duration)
            if result.sheet_duration is not None
            else None,
        },
        "aggregate": {
            # The maximum, not the mean: a three-second artifact in a ten-minute
            # video must not be averaged away by the thirty clean windows
            # around it.  Kept beside mean_score so the dilution the maximum
            # avoids stays visible.
            "flicker_score": scalar(result.flicker_score),
            "mean_score": scalar(result.mean_score),
            "severity_band": result.severity_band,
            "route": result.route,
            "confidence": scalar(result.confidence),
            "total_windows": int(result.total_windows),
            "positive_windows": int(result.positive_windows),
            "worst_window_index": int(result.worst_index),
            "worst_segment_start": scalar(result.worst_segment[0]),
            "worst_segment_end": scalar(result.worst_segment[1]),
            "band_counts": band_counts(result.windows),
            "scores": scores,
            "horizontal_coherence": scalar(result.horizontal_coherence),
            "valid_fraction": scalar(result.valid_fraction),
        },
        "windows": [window_to_dict(window) for window in result.windows],
    }


def window_to_dict(window: WindowRecord) -> dict[str, Any]:
    """Serialize one window, splitting evidence from diagnostics."""
    return {
        "index": int(window.index),
        "start_time": scalar(window.start_time),
        "end_time": scalar(window.end_time),
        "fps": scalar(window.fps),
        "frame_count": int(window.frame_count),
        "score": scalar(window.score),
        "severity_band": window.severity_band,
        "route": window.route,
        "confidence": scalar(window.confidence),
        "is_worst": bool(window.is_worst),
        "scores": {
            _detector_key(name): scalar(score) for name, score in window.detector_scores.items()
        },
        "diagnostics": {
            "valid_fraction": scalar(window.valid_fraction),
            "horizontal_coherence": scalar(window.horizontal_coherence),
        },
        "measurements": window.measurements,
    }


def error_record(
    key: str,
    error: str,
    detector_version: str,
    *,
    completed_at: str,
) -> dict[str, Any]:
    """Build a stable record for a video that could not be processed.

    Windows are absent rather than zero-filled: no measurement was taken, and a
    zero would read as one.  ``--resume`` retries exactly these.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": error,
        "video_key": key,
        "detector_version": detector_version,
        "completed_at": completed_at,
        "processing_time_seconds": None,
        "source": {},
        "aggregate": {},
        "windows": [],
    }


def band_counts(windows: list[WindowRecord]) -> dict[str, int]:
    """Count windows per severity band, the histogram a maximum cannot give."""
    counts = {"none": 0, "mild": 0, "extreme": 0}
    for window in windows:
        counts[window.severity_band] = counts.get(window.severity_band, 0) + 1
    return counts


def video_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a record into the one-row-per-video rollup schema."""
    aggregate = record.get("aggregate") or {}
    scores = aggregate.get("scores") or {}
    counts = aggregate.get("band_counts") or {}
    source = record.get("source") or {}
    if record.get("status") != "ok":
        return {
            **dict.fromkeys(VIDEO_FIELDS, ""),
            "schema_version": record.get("schema_version", SCHEMA_VERSION),
            "status": record.get("status", "error"),
            "error": record.get("error", ""),
            "video_key": record.get("video_key", ""),
            "detector_version": record.get("detector_version", ""),
            "completed_at": record.get("completed_at", ""),
        }
    return {
        "schema_version": record.get("schema_version", SCHEMA_VERSION),
        "status": "ok",
        "error": "",
        "video_key": record.get("video_key", ""),
        "flicker_score": aggregate.get("flicker_score", ""),
        "severity_band": aggregate.get("severity_band", ""),
        "route": aggregate.get("route", ""),
        "confidence": aggregate.get("confidence", ""),
        "worst_segment_start": aggregate.get("worst_segment_start", ""),
        "worst_segment_end": aggregate.get("worst_segment_end", ""),
        "illuminant_score": scores.get("illuminant", 0.0),
        "rolling_band_score": scores.get("rolling_band", 0.0),
        "awb_score": scores.get("awb", 0.0),
        "horizontal_coherence": aggregate.get("horizontal_coherence", ""),
        "valid_fraction": aggregate.get("valid_fraction", ""),
        "processing_time_seconds": record.get("processing_time_seconds", ""),
        "detector_version": record.get("detector_version", ""),
        "completed_at": record.get("completed_at", ""),
        "total_windows": aggregate.get("total_windows", ""),
        "positive_windows": aggregate.get("positive_windows", ""),
        "mean_score": aggregate.get("mean_score", ""),
        "worst_window_index": aggregate.get("worst_window_index", ""),
        "windows_mild": counts.get("mild", ""),
        "windows_extreme": counts.get("extreme", ""),
        "duration_seconds": source.get("duration", ""),
        "project_name": source.get("project_name", ""),
        "video_id": source.get("video_id", ""),
    }


def window_rows(record: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Flatten a record into one row per window.

    Failed videos yield nothing: a window row asserts that a window was
    measured.  The video table is where a failure is accounted for.
    """
    if record.get("status") != "ok":
        return
    aggregate = record.get("aggregate") or {}
    context = {
        "schema_version": record.get("schema_version", SCHEMA_VERSION),
        "video_key": record.get("video_key", ""),
        "project_name": (record.get("source") or {}).get("project_name", ""),
        "video_flicker_score": aggregate.get("flicker_score", ""),
        "video_severity_band": aggregate.get("severity_band", ""),
        "video_route": aggregate.get("route", ""),
        "total_windows": aggregate.get("total_windows", ""),
        "detector_version": record.get("detector_version", ""),
        "completed_at": record.get("completed_at", ""),
    }
    for window in record.get("windows") or []:
        scores = window.get("scores") or {}
        diagnostics = window.get("diagnostics") or {}
        measurements = window.get("measurements") or {}
        row = {
            **context,
            "window_index": window.get("index", ""),
            "start_time": window.get("start_time", ""),
            "end_time": window.get("end_time", ""),
            "fps": window.get("fps", ""),
            "frame_count": window.get("frame_count", ""),
            "window_score": window.get("score", ""),
            "severity_band": window.get("severity_band", ""),
            "route": window.get("route", ""),
            "confidence": window.get("confidence", ""),
            "is_worst": window.get("is_worst", ""),
            "illuminant_score": scores.get("illuminant", ""),
            "rolling_band_score": scores.get("rolling_band", ""),
            "awb_score": scores.get("awb", ""),
            "valid_fraction": diagnostics.get("valid_fraction", ""),
            "horizontal_coherence": diagnostics.get("horizontal_coherence", ""),
        }
        for key, metrics_type in _TABULATED_METRICS:
            measured = measurements.get(key) or {}
            for field in fields(metrics_type):
                row[f"{key}_{field.name}"] = measured.get(field.name, "")
        # Projected through the declared schema, so the row is complete and in
        # column order whatever the record happened to contain.
        yield {field: row.get(field, "") for field in WINDOW_FIELDS}
