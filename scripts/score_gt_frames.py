#!/usr/bin/env python3
"""Score every BurstFlicker-G ground-truth frame with the calibrated detector.

The ``gt/`` half of BurstFlicker-G is flicker-free by construction, so this is a
false-positive audit: every row carries a label of ``none``, and any row whose
``severity_band`` is not ``none`` is the detector inventing flicker that is not
there.  Nothing here re-fits anything.  The weights and normalization scales are
read from ``configs/detector.yaml``, so this measures the calibration as
committed rather than a private copy of it.

Granularity.  The detectors are *window* measurements -- a Welch PSD over the
window's luminance, a per-row phase ramp across its frames -- and a BurstFlicker
burst is 10 frames at 30 fps, one 0.33 s window.  A frame in isolation therefore
has no spectrum and no score.  What each row carries instead is:

* the clip's ``flicker_score`` and per-detector scores, which are properties of
  the window the frame belongs to, repeated across that window's frames, and
* seven ``frame_*`` columns that *are* per-frame: this frame's luminance, chroma,
  and row-profile residual, defined so that two of them reduce exactly to the
  window measurement they underpin (see :class:`FrameSignals`).

So the score columns answer "what did the model say about the burst containing
this frame" and the ``frame_*`` columns answer "what did this frame contribute".
Conflating the two would be the only dishonest way to fill the file.

Scoring reproduces ``main.process_video`` window for window under whatever
calibration the config currently holds, so a run is comparable with production
by construction rather than by convention.  Verified against
``output/burstflicker/g_scores_prodcfg.csv``, which was produced at weights
0.30/0.50/0.20: every score and diagnostic column matched to the last digit
across all 369 clips.  Change the weights and the scores move, so read the
``weight_*`` columns before comparing two runs.

Usage:
    python scripts/score_gt_frames.py
    python scripts/score_gt_frames.py --limit 20 --workers 4
    python scripts/score_gt_frames.py --splits test --output output/gt_test_frames.csv
    python scripts/score_gt_frames.py --region full-frame     # no fisheye masking
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

import av
import numpy as np
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.calibration.thresholds import (  # noqa: E402
    AWBNormalizer,
    IlluminantNormalizer,
    RollingBandNormalizer,
)
from src.core.aggregator import DetectionAggregator, ScoreNormalizer, WindowMetrics  # noqa: E402
from src.core.pipeline import DetectionPipeline  # noqa: E402
from src.core.types import SignalFeatures  # noqa: E402
from src.data.reader import VideoReader  # noqa: E402
from src.data.sampler import WindowSampler, WindowSpec  # noqa: E402
from src.detectors.awb import AWBDetector, AWBMetrics  # noqa: E402
from src.detectors.base import BaseDetector  # noqa: E402
from src.detectors.illuminant import IlluminantDetector, IlluminantMetrics  # noqa: E402
from src.detectors.rolling_band import RollingBandDetector, RollingBandMetrics  # noqa: E402
from src.features.extractor import FeatureExtractor  # noqa: E402
from src.signals.extractor import SignalExtractor  # noqa: E402
from src.signals.mask import RegionPolicy  # noqa: E402

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
DEFAULT_DATASET_ROOT = (
    "data/kagglehub/datasets/lishenqu/burstflicker/versions/1/BurstFlicker/BurstFlicker-G"
)
DEFAULT_OUTPUT = "output/burstflicker/gt_frame_scores.csv"
# BurstFlicker-G ``gt/`` clips are the flicker-free reference of each pair.
GROUND_TRUTH_LABEL = "none"

# Which pixels to measure over.  See :func:`load_config`.
REGION_MASKED = "masked"
REGION_NO_BOTTOM_EXCLUSION = "no-bottom-exclusion"
REGION_FULL_FRAME = "full-frame"
REGION_POLICIES = (REGION_MASKED, REGION_NO_BOTTOM_EXCLUSION, REGION_FULL_FRAME)

# Pool sizing.  See :func:`safe_worker_count`.
_FRAMES_IN_FLIGHT_PER_WORKER = 4
_MEMORY_BUDGET_FRACTION = 0.20

OUTPUT_FIELDS = (
    # Identity and decode facts.
    "schema_version",
    "status",
    "error",
    "clip_key",
    "split",
    "scene",
    "source_path",
    "n_frames",
    "fps",
    "native_width",
    "native_height",
    "frame_index",
    "frame_time_s",
    # Clip-level verdict: the maximum window score, as in production.
    "flicker_score",
    "severity_band",
    "route",
    "confidence",
    "label",
    "false_positive",
    # The window this frame belongs to, and its normalized detector evidence.
    "window_index",
    "window_start_s",
    "window_end_s",
    "window_score",
    "illuminant_score",
    "rolling_band_score",
    "awb_score",
    # Raw detector measurements for that window: why the scores came out as they did.
    "dominant_frequency_hz",
    "peak_prominence",
    "peak_to_median_ratio",
    "modulation_depth",
    "band_amplitude",
    "phase_linearity",
    "band_cycles_per_frame",
    "drift_velocity_rows_per_s",
    "horizontal_coherence",
    "chroma_prominence",
    "chroma_periodicity",
    "chroma_variance",
    "luma_chroma_decorrelation",
    "ae_prominence",
    "ae_periodicity",
    "ae_dominant_frequency_hz",
    "ae_in_band",
    "valid_fraction",
    # Genuinely per-frame signals.
    "frame_luma",
    "frame_luma_modulation",
    "frame_chroma_a",
    "frame_chroma_b",
    "frame_chroma_deviation",
    "frame_band_residual",
    "frame_band_projection",
    # Provenance: which calibration produced the row.
    "weight_illuminant",
    "weight_rolling_band",
    "weight_awb",
    "region_policy",
    "detector_version",
    "processing_time_seconds",
)


@dataclass(frozen=True, slots=True)
class GroundTruthClip:
    """One flicker-free clip to score, addressed by a repo-relative path."""

    key: str
    split: str
    scene: str
    path: Path


@dataclass(slots=True)
class FrameSignals:
    """Per-frame signal values for the frames of one window.

    Each array holds one value per frame, and the first two are defined so that
    their relationship to the window measurement they underpin is checkable
    rather than merely suggestive.  Both reduce to within float32 precision --
    around 1e-5 relative -- because the detectors measure in float32 while these
    are accumulated in float64:

    ``luma_modulation``
        Linearly detrended luminance divided by the window's mean level, signed.
        Its RMS over the window's frames is the window's ``modulation_depth``.
    ``chroma_deviation``
        Chroma magnitude minus its window mean.  Its standard deviation over the
        window's frames is the window's ``chroma_variance``.
    ``band_residual``
        RMS across rows of the frame's row profile after removing each row's
        temporal mean and then the frame's own spatial mean, divided by scene
        level -- so it is scale-free and blind to a brightness change that moves
        every row together.  This is the per-frame *analogue* of
        ``band_amplitude``, not a decomposition of it: ``band_amplitude`` is
        measured only at the flicker frequency, and a single frame cannot carry a
        frequency.  Expect it to run larger than ``band_amplitude``, since it
        also contains the broadband spatial noise the frequency projection
        rejects.
    ``band_projection``
        The same residual projected onto the spatial frequency the detector
        found, divided by rows times level.  Spatially frequency-selective, so it
        isolates structure shaped like the detected band; still broadband in
        time.  Zero when the detector found no periodic component to measure a
        spatial frequency at.
    """

    luma: np.ndarray
    luma_modulation: np.ndarray
    chroma_a: np.ndarray
    chroma_b: np.ndarray
    chroma_deviation: np.ndarray
    band_residual: np.ndarray
    band_projection: np.ndarray


@dataclass(slots=True)
class ScoredWindow:
    """One window's frames, evidence, and raw measurements."""

    index: int
    spec: WindowSpec
    first_frame_index: int
    n_frames: int
    metrics: WindowMetrics
    signals: FrameSignals
    measurements: dict[str, object]


# ----------------------------------------------------------------------
# Clip discovery
# ----------------------------------------------------------------------


def discover_clips(dataset_root: Path, splits: Sequence[str]) -> list[GroundTruthClip]:
    """Find every ``<split>/gt/*.mp4`` clip under a BurstFlicker-G root.

    The directory is globbed rather than read from ``data/burstflicker/g_manifest.csv``
    because that manifest interleaves ``flicker/`` and ``gt/`` entries and is a
    snapshot; the tree is what will actually be decoded.
    """
    clips: list[GroundTruthClip] = []
    for split in splits:
        gt_directory = dataset_root / split / "gt"
        if not gt_directory.is_dir():
            raise FileNotFoundError(f"No ground-truth directory at {gt_directory}")
        for path in sorted(gt_directory.glob("*.mp4")):
            clips.append(
                GroundTruthClip(
                    key=f"{split}/gt/{path.stem}",
                    split=split,
                    scene=path.stem,
                    path=path,
                )
            )
    if not clips:
        raise FileNotFoundError(f"No .mp4 clips found under {dataset_root} for splits {splits}")
    return clips


# ----------------------------------------------------------------------
# Detection components
# ----------------------------------------------------------------------


def build_detection_components(
    config: dict[str, Any],
) -> tuple[DetectionPipeline, DetectionAggregator]:
    """Construct the enabled detectors and the configured aggregation policy.

    Deliberately a copy of ``main.build_detection_components`` rather than an
    import of it: importing ``main`` pulls in the S3 source layer and its boto3
    dependency, which this offline script has no use for.
    """
    aggregation_config = config["aggregation"]
    detectors: list[BaseDetector] = []
    normalizers: dict[str, ScoreNormalizer] = {}

    if config["illuminant_detector"]["enabled"]:
        detectors.append(IlluminantDetector())
        normalizers["IlluminantDetector"] = IlluminantNormalizer(
            **aggregation_config["illuminant_normalization"]
        )
    if config["rolling_band_detector"]["enabled"]:
        detectors.append(
            RollingBandDetector(smoothing_sigma=config["rolling_band_detector"]["smoothing_sigma"])
        )
        normalizers["RollingBandDetector"] = RollingBandNormalizer(
            **aggregation_config["rolling_band_normalization"]
        )
    awb_config = config.get("awb_detector", {"enabled": False})
    if awb_config.get("enabled", False):
        detectors.append(
            AWBDetector(
                ae_min_frequency=awb_config.get("ae_min_frequency", 0.5),
                ae_max_frequency=awb_config.get("ae_max_frequency", 5.0),
            )
        )
        normalizers["AWBDetector"] = AWBNormalizer(**aggregation_config["awb_normalization"])

    aggregator = DetectionAggregator(
        weights=aggregation_config["weights"],
        normalizers=normalizers,
        positive_threshold=aggregation_config["positive_threshold"],
    )
    return DetectionPipeline(detectors), aggregator


def classify_score(
    score: float,
    *,
    mild_threshold: float,
    extreme_threshold: float,
) -> tuple[str, str, float]:
    """Map a normalized score to a band, routing decision, and boundary margin."""
    if not 0.0 <= mild_threshold < extreme_threshold <= 1.0:
        raise ValueError("decision thresholds must satisfy 0 <= mild < extreme <= 1")
    if score < mild_threshold:
        band, route = "none", "accept"
    elif score < extreme_threshold:
        band, route = "mild", "review"
    else:
        band, route = "extreme", "reject"

    nearest_boundary = min(abs(score - mild_threshold), abs(score - extreme_threshold))
    confidence = min(nearest_boundary / max(extreme_threshold - mild_threshold, 1e-12), 1.0)
    return band, route, confidence


# ----------------------------------------------------------------------
# Per-frame signals
# ----------------------------------------------------------------------


def frame_signals(
    signals: SignalFeatures,
    band: RollingBandMetrics,
    smoothing_sigma: float,
) -> FrameSignals:
    """Reduce a window's signal arrays to one value per frame.

    See :class:`FrameSignals` for each definition and the window measurement it
    reduces to.  Row profiles are smoothed with the same sigma the rolling-band
    detector uses, and scene level is that array's mean, so ``band_residual``
    and ``band_amplitude`` are divided by the same denominator.
    """
    luma = np.asarray(signals.luma, dtype=np.float64)
    chroma_a = np.asarray(signals.chroma_a, dtype=np.float64)
    chroma_b = np.asarray(signals.chroma_b, dtype=np.float64)
    chroma_magnitude = np.sqrt(chroma_a**2 + chroma_b**2)

    luma_level = float(np.mean(np.abs(luma))) if luma.size else 0.0
    if luma.size >= 2 and luma_level > np.finfo(np.float32).eps:
        luma_modulation = np.asarray(detrend(luma), dtype=np.float64) / luma_level
    else:
        luma_modulation = np.zeros_like(luma)

    chroma_deviation = (
        chroma_magnitude - float(np.mean(chroma_magnitude))
        if chroma_magnitude.size
        else chroma_magnitude
    )

    band_residual, band_projection = _band_signals(
        np.asarray(signals.row_profiles, dtype=np.float64),
        band.band_cycles_per_frame,
        smoothing_sigma,
    )
    return FrameSignals(
        luma=luma,
        luma_modulation=luma_modulation,
        chroma_a=chroma_a,
        chroma_b=chroma_b,
        chroma_deviation=chroma_deviation,
        band_residual=band_residual,
        band_projection=band_projection,
    )


def _band_signals(
    row_profiles: np.ndarray,
    band_cycles_per_frame: float,
    smoothing_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each frame's row-profile residual and its projection onto the band.

    The two subtractions mirror the detector.  Removing each row's temporal mean
    discards the static row pattern -- frame geometry and scene layout, which are
    not flicker.  Removing each frame's spatial mean then discards a brightness
    change that moves every row in lockstep, which is global flicker and belongs
    to the illuminant channel, not to banding.  What survives is horizontal
    structure that appeared in this frame and not in the window's average.

    Only the projection's magnitude is used, so the sign the detector's spatial
    frequency lost does not matter: for a real profile the two conjugate
    frequencies have identical magnitude.
    """
    frames = row_profiles.shape[0] if row_profiles.ndim == 2 else 0
    rows = row_profiles.shape[1] if row_profiles.ndim == 2 else 0
    if frames < 1 or rows < 1:
        return np.zeros(frames, dtype=np.float64), np.zeros(frames, dtype=np.float64)

    smoothed = gaussian_filter1d(row_profiles, sigma=smoothing_sigma, axis=1)
    level = float(np.mean(smoothed))
    if level <= np.finfo(np.float32).eps:
        return np.zeros(frames, dtype=np.float64), np.zeros(frames, dtype=np.float64)

    centered = smoothed - smoothed.mean(axis=0, keepdims=True)
    structured = centered - centered.mean(axis=1, keepdims=True)

    residual = np.sqrt(np.mean(structured**2, axis=1)) / level

    cycles_per_row = band_cycles_per_frame / rows
    if cycles_per_row <= 0.0:
        # No periodic component means the detector never estimated a spatial
        # frequency, so there is no band shape to project onto.
        return residual, np.zeros(frames, dtype=np.float64)

    kernel = np.exp(-2j * np.pi * cycles_per_row * np.arange(rows, dtype=np.float64))
    projection = np.abs(structured @ kernel) / (rows * level)
    return residual, projection


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def score_clip(clip: GroundTruthClip, config: dict[str, Any]) -> list[dict[str, object]]:
    """Score one clip and return one row per decoded frame.

    Windows follow the configured sampling schedule, which for a 0.33 s burst is
    a single window covering the whole clip.  A frame covered by more than one
    window -- possible only on longer footage than BurstFlicker supplies -- is
    attributed to the highest-scoring window that contains it, matching the
    ``max`` the clip-level score already uses.
    """
    started_at = perf_counter()
    sampler_config = config["sampling"]
    decision_config = config["decision"]
    detector_pipeline, aggregator = build_detection_components(config)
    sampler = WindowSampler(
        window_duration=sampler_config["window_duration"],
        stride=sampler_config["stride"],
    )
    resize = _parse_resize(sampler_config.get("resize"))
    region_policy = RegionPolicy.from_config(config.get("analysis_region"))
    smoothing_sigma = float(config["rolling_band_detector"]["smoothing_sigma"])
    flicker_band = config.get("flicker_band") or {}
    min_flicker = float(flicker_band.get("min_frequency") or 0.0)
    max_flicker_value = flicker_band.get("max_frequency")
    max_flicker = float(max_flicker_value) if max_flicker_value is not None else None

    scored: list[ScoredWindow] = []
    with VideoReader(clip.path) as reader:
        metadata = reader.metadata()
        for index, spec in enumerate(sampler.sample(metadata.duration)):
            window = reader.read_window(
                start=spec.start_time,
                duration=spec.end_time - spec.start_time,
                resize=resize,
            )
            if not len(window.frames):
                logger.warning("Empty window %d in %s; skipping", index, clip.key)
                continue

            signals = SignalExtractor.extract(window, region_policy)
            features = FeatureExtractor.extract(
                signals,
                window.fps,
                min_flicker_frequency=min_flicker,
                max_flicker_frequency=max_flicker,
            )
            results = detector_pipeline.run(features)
            scored.append(
                ScoredWindow(
                    index=index,
                    spec=spec,
                    # Constant frame rate is assumed here, which holds for every
                    # BurstFlicker clip; on variable-rate footage the index is
                    # nominal and ``frame_time_s`` should be read as such.
                    first_frame_index=int(round(spec.start_time * window.fps)),
                    n_frames=len(window.frames),
                    metrics=aggregator.aggregate(results),
                    signals=frame_signals(
                        signals,
                        _rolling_band_metrics(results),
                        smoothing_sigma,
                    ),
                    measurements=_raw_measurements(results, features.valid_fraction),
                )
            )

    if not scored:
        raise ValueError(f"No decodable frames in {clip.path}")

    video_metrics = aggregator.aggregate_video([window.metrics for window in scored])
    severity_band, route, confidence = classify_score(
        video_metrics.max_score,
        mild_threshold=decision_config["mild_threshold"],
        extreme_threshold=decision_config["extreme_threshold"],
    )
    clip_context = {
        "clip_key": clip.key,
        "split": clip.split,
        "scene": clip.scene,
        "source_path": _repo_relative(clip.path),
        "n_frames": sum(window.n_frames for window in scored),
        "fps": metadata.fps,
        "native_width": metadata.width,
        "native_height": metadata.height,
        "flicker_score": video_metrics.max_score,
        "severity_band": severity_band,
        "route": route,
        "confidence": confidence,
        "label": GROUND_TRUTH_LABEL,
        "false_positive": int(severity_band != GROUND_TRUTH_LABEL),
        "processing_time_seconds": perf_counter() - started_at,
    }
    rows = _frame_rows(scored, clip_context, config)
    logger.info(
        "Scored %s: flicker_score=%.4f band=%s frames=%d",
        clip.key,
        video_metrics.max_score,
        severity_band,
        len(rows),
    )
    return rows


def _frame_rows(
    scored: Sequence[ScoredWindow],
    clip_context: dict[str, object],
    config: dict[str, Any],
) -> list[dict[str, object]]:
    """Emit one row per frame, attributing each to its best-scoring window."""
    provenance = _provenance(config)
    best_by_frame: dict[int, tuple[ScoredWindow, int]] = {}
    for window in scored:
        for offset in range(window.n_frames):
            frame_index = window.first_frame_index + offset
            incumbent = best_by_frame.get(frame_index)
            if incumbent is None or window.metrics.score > incumbent[0].metrics.score:
                best_by_frame[frame_index] = (window, offset)

    rows: list[dict[str, object]] = []
    fps = float(clip_context["fps"])  # type: ignore[arg-type]
    for frame_index in sorted(best_by_frame):
        window, offset = best_by_frame[frame_index]
        signals = window.signals
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "error": "",
                **clip_context,
                "frame_index": frame_index,
                "frame_time_s": frame_index / fps if fps > 0 else "",
                "window_index": window.index,
                "window_start_s": window.spec.start_time,
                "window_end_s": window.spec.end_time,
                "window_score": window.metrics.score,
                "illuminant_score": window.metrics.detector_scores.get("IlluminantDetector", 0.0),
                "rolling_band_score": window.metrics.detector_scores.get(
                    "RollingBandDetector", 0.0
                ),
                "awb_score": window.metrics.detector_scores.get("AWBDetector", 0.0),
                **window.measurements,
                "frame_luma": _at(signals.luma, offset),
                "frame_luma_modulation": _at(signals.luma_modulation, offset),
                "frame_chroma_a": _at(signals.chroma_a, offset),
                "frame_chroma_b": _at(signals.chroma_b, offset),
                "frame_chroma_deviation": _at(signals.chroma_deviation, offset),
                "frame_band_residual": _at(signals.band_residual, offset),
                "frame_band_projection": _at(signals.band_projection, offset),
                **provenance,
            }
        )
    return rows


def _raw_measurements(
    results: dict[str, object],
    valid_fraction: float,
) -> dict[str, object]:
    """Flatten the detectors' raw metrics, defaulting a disabled detector to blank."""
    illuminant = results.get("IlluminantDetector")
    band = results.get("RollingBandDetector")
    awb = results.get("AWBDetector")
    measurements: dict[str, object] = {
        "dominant_frequency_hz": "",
        "peak_prominence": "",
        "peak_to_median_ratio": "",
        "modulation_depth": "",
        "band_amplitude": "",
        "phase_linearity": "",
        "band_cycles_per_frame": "",
        "drift_velocity_rows_per_s": "",
        "horizontal_coherence": "",
        "chroma_prominence": "",
        "chroma_periodicity": "",
        "chroma_variance": "",
        "luma_chroma_decorrelation": "",
        "ae_prominence": "",
        "ae_periodicity": "",
        "ae_dominant_frequency_hz": "",
        "ae_in_band": "",
        "valid_fraction": valid_fraction,
    }
    if isinstance(illuminant, IlluminantMetrics):
        measurements.update(
            dominant_frequency_hz=illuminant.dominant_frequency,
            peak_prominence=illuminant.peak_prominence,
            peak_to_median_ratio=illuminant.peak_to_median_ratio,
            modulation_depth=illuminant.modulation_depth,
        )
    if isinstance(band, RollingBandMetrics):
        measurements.update(
            band_amplitude=band.band_amplitude,
            phase_linearity=band.phase_linearity,
            band_cycles_per_frame=band.band_cycles_per_frame,
            drift_velocity_rows_per_s=band.drift_velocity,
            horizontal_coherence=band.horizontal_coherence,
        )
    if isinstance(awb, AWBMetrics):
        measurements.update(
            chroma_prominence=awb.chroma_prominence,
            chroma_periodicity=awb.chroma_periodicity,
            chroma_variance=awb.chroma_variance,
            luma_chroma_decorrelation=awb.luma_chroma_decorrelation,
            ae_prominence=awb.ae_prominence,
            ae_periodicity=awb.ae_periodicity,
            ae_dominant_frequency_hz=awb.ae_dominant_frequency,
            ae_in_band=int(awb.ae_in_band),
        )
    return measurements


def _rolling_band_metrics(results: dict[str, object]) -> RollingBandMetrics:
    """Return the rolling-band result, or an inert stand-in if it is disabled."""
    metrics = results.get("RollingBandDetector")
    if isinstance(metrics, RollingBandMetrics):
        return metrics
    return RollingBandMetrics(
        band_amplitude=0.0,
        phase_linearity=0.0,
        band_cycles_per_frame=0.0,
        drift_velocity=0.0,
    )


def _provenance(config: dict[str, Any]) -> dict[str, object]:
    """Record which calibration produced a row, so two runs stay distinguishable.

    ``region_policy`` is derived from the config rather than passed down from the
    CLI, because a worker receives only the config and the two must not be able
    to disagree about what was measured.
    """
    weights = config["aggregation"]["weights"]
    region = config.get("analysis_region") or {}
    excluded = float(region.get("exclude_bottom_fraction", 0.0))
    border = float(region.get("border_threshold", 0.0))
    if excluded > 0.0:
        policy = REGION_MASKED
    elif border > 0.0:
        policy = REGION_NO_BOTTOM_EXCLUSION
    else:
        policy = REGION_FULL_FRAME
    return {
        "weight_illuminant": weights.get("IlluminantDetector", 0.0),
        "weight_rolling_band": weights.get("RollingBandDetector", 0.0),
        "weight_awb": weights.get("AWBDetector", 0.0),
        "region_policy": policy,
        "detector_version": config.get("detector_version", ""),
    }


def _at(values: np.ndarray, index: int) -> object:
    return float(values[index]) if index < values.size else ""


def safe_worker_count(requested: int, sample_path: Path) -> int:
    """Cap workers so concurrent native-resolution decode fits in RAM.

    PyAV decodes at the source resolution and only then reformats to the
    analysis size, so peak memory scales with the *source* frame rather than the
    320x180 that gets measured.  BurstFlicker-G is 6960x4640, which is 97 MB per
    RGB frame; eight workers of that on an 8 GB machine swaps instead of
    computing, and the pool wedges rather than failing cleanly.  Sizing the pool
    from one probed clip turns that into a logged cap.
    """
    try:
        with av.open(str(sample_path)) as container:
            stream = container.streams.video[0]
            frame_bytes = int(stream.width) * int(stream.height) * 3
        total_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError, IndexError):
        # Unknown resolution or no sysconf: leave the caller's choice alone
        # rather than guessing a cap from nothing.
        return requested

    per_worker = frame_bytes * _FRAMES_IN_FLIGHT_PER_WORKER
    allowed = max(1, int(total_bytes * _MEMORY_BUDGET_FRACTION) // max(per_worker, 1))
    if allowed < requested:
        logger.warning(
            "Capping workers %d -> %d: %dx%d decode needs ~%.0f MB per worker, "
            "and %.1f GB of RAM allows only that many within a %.0f%% budget",
            requested,
            allowed,
            stream.width,
            stream.height,
            per_worker / 1e6,
            total_bytes / 1e9,
            _MEMORY_BUDGET_FRACTION * 100,
        )
    return min(requested, allowed)


# ----------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------


def run(
    clips: Sequence[GroundTruthClip],
    output_path: Path,
    config: dict[str, Any],
    *,
    workers: int,
    resume: bool,
) -> None:
    """Score every clip in parallel, flushing rows as each clip completes."""
    retained = _successful_rows(output_path) if resume else {}
    if retained:
        logger.info("Resuming: %d clips already scored in %s", len(retained), output_path)
    pending = [clip for clip in clips if clip.key not in retained]
    if not pending:
        logger.info("Nothing to score; %s is already complete", output_path)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    scores: dict[str, float] = {
        key: float(str(rows[0]["flicker_score"]) or 0.0) for key, rows in retained.items()
    }
    bands: dict[str, str] = {key: str(rows[0]["severity_band"]) for key, rows in retained.items()}
    frame_count = sum(len(rows) for rows in retained.values())
    failed: list[str] = []

    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for rows in retained.values():
            writer.writerows(rows)
        output_file.flush()

        for clip, rows in _scored_in_parallel(pending, config, workers):
            if rows and rows[0].get("status") == "ok":
                scores[clip.key] = float(rows[0]["flicker_score"])  # type: ignore[arg-type]
                bands[clip.key] = str(rows[0]["severity_band"])
                frame_count += len(rows)
            else:
                failed.append(clip.key)
            writer.writerows(rows)
            output_file.flush()

    elapsed = perf_counter() - started_at
    # Rows are written in worker-completion order so a killed run still resumes
    # from whatever finished.  Sorting once at the end, only on a clean finish,
    # keeps that property while making the delivered file byte-reproducible.
    _sort_output(output_path)
    logger.info(
        "Finished: %d clips, %d frame rows, %d failed, %.1fs -> %s",
        len(scores) + len(failed),
        frame_count,
        len(failed),
        elapsed,
        output_path,
    )
    _print_summary(scores, bands, frame_count, failed, output_path)


def _sort_output(output_path: Path) -> None:
    """Rewrite the output in ``(clip_key, frame_index)`` order."""
    with output_path.open(newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))

    def sort_key(row: dict[str, str]) -> tuple[str, int]:
        index = row.get("frame_index") or ""
        # An error row carries no frame index and sorts ahead of its clip's rows,
        # of which there are none.
        return row.get("clip_key", ""), int(index) if index.isdigit() else -1

    rows.sort(key=sort_key)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _scored_in_parallel(
    clips: Sequence[GroundTruthClip],
    config: dict[str, Any],
    workers: int,
) -> Iterable[tuple[GroundTruthClip, list[dict[str, object]]]]:
    """Yield each clip's rows as its worker finishes, in completion order."""
    if workers <= 1:
        for clip in clips:
            yield clip, _score_or_error_row(clip, config)
        return

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_score_or_error_row, clip, config): clip for clip in clips}
        for completed, future in enumerate(as_completed(futures), start=1):
            clip = futures[future]
            try:
                rows = future.result()
            except Exception as error:  # pragma: no cover - worker crash
                logger.exception("Worker exception for %s", clip.key)
                rows = [_error_row(clip, error, config)]
            if completed % 25 == 0:
                logger.info("Progress: %d/%d clips", completed, len(clips))
            yield clip, rows


def _score_or_error_row(
    clip: GroundTruthClip,
    config: dict[str, Any],
) -> list[dict[str, object]]:
    """Score one clip inside a worker, turning a failure into a single row."""
    try:
        return score_clip(clip, config)
    except Exception as error:
        logger.exception("Failed clip: %s", clip.key)
        return [_error_row(clip, error, config)]


def _error_row(
    clip: GroundTruthClip,
    error: Exception,
    config: dict[str, Any],
) -> dict[str, object]:
    """One blank-scored row standing in for a clip that could not be decoded."""
    row: dict[str, object] = dict.fromkeys(OUTPUT_FIELDS, "")
    row.update(
        schema_version=SCHEMA_VERSION,
        status="error",
        error=str(error),
        clip_key=clip.key,
        split=clip.split,
        scene=clip.scene,
        source_path=_repo_relative(clip.path),
        label=GROUND_TRUTH_LABEL,
        **_provenance(config),
    )
    return row


def _successful_rows(output_path: Path) -> dict[str, list[dict[str, object]]]:
    """Return an existing run's successful rows, grouped by clip.

    A clip is retained only if every one of its rows is ``ok``, so a clip whose
    write was interrupted part-way is rescored rather than silently kept short.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return {}
    grouped: dict[str, list[dict[str, object]]] = {}
    healthy: dict[str, bool] = {}
    with output_path.open(newline="", encoding="utf-8") as output_file:
        for row in csv.DictReader(output_file):
            key = (row.get("clip_key") or "").strip()
            if not key:
                continue
            grouped.setdefault(key, []).append(
                {field: row.get(field, "") for field in OUTPUT_FIELDS}
            )
            healthy[key] = healthy.get(key, True) and row.get("status") == "ok"
    return {key: rows for key, rows in grouped.items() if healthy.get(key)}


def _print_summary(
    scores: dict[str, float],
    bands: dict[str, str],
    frame_count: int,
    failed: Sequence[str],
    output_path: Path,
) -> None:
    """Print the false-positive picture the run exists to produce."""
    print(f"\n{'=' * 66}")
    print("  BURSTFLICKER-G GROUND TRUTH (flicker-free) -- SCORE DISTRIBUTION")
    print(f"{'=' * 66}")
    print(f"  Output:        {output_path}")
    print(f"  Clips scored:  {len(scores)}      Frame rows: {frame_count}")
    if failed:
        print(f"  FAILED:        {len(failed)} clips -> {', '.join(failed[:5])}")
    if not scores:
        print(f"{'=' * 66}\n")
        return

    values = sorted(scores.values())
    print(
        "  flicker_score: "
        f"min={values[0]:.4f}  median={median(values):.4f}  "
        f"mean={sum(values) / len(values):.4f}  "
        f"p95={values[min(int(0.95 * len(values)), len(values) - 1)]:.4f}  "
        f"max={values[-1]:.4f}"
    )
    print("\n  Severity band (every clip is labelled 'none'):")
    for band in ("none", "mild", "extreme"):
        count = sum(1 for value in bands.values() if value == band)
        marker = "" if band == "none" else "   <-- false positive"
        print(f"    {band:>8}: {count:>4} clips ({count / len(bands):>6.1%}){marker}")
    worst = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    print("\n  Highest-scoring clean clips:")
    for key, value in worst:
        print(f"    {value:.4f}  {key}")
    print(f"{'=' * 66}\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def load_config(config_path: Path, *, region: str) -> dict[str, Any]:
    """Load the committed detector settings under one of three region policies.

    The choice exists because the committed ``analysis_region`` block is fisheye
    policy: it drops the bottom 25% of the frame as the wearer's torso and treats
    dark pixels as matte border.  BurstFlicker is neither egocentric nor
    pillarboxed, so that exclusion discards a quarter of usable rows.  The
    committed policy is still the default, so a run stays comparable with the
    production corpus, and the other two reproduce the existing comparison files:

    ``masked``
        The config untouched.  Reproduces ``g_scores_prodcfg.csv``.
    ``no-bottom-exclusion``
        Border detection kept, torso exclusion dropped.  Reproduces
        ``g_scores_fullframe.csv``.
    ``full-frame``
        Every pixel measured.  ``min_valid_fraction`` of 1.0 forces the region
        detector down its own full-frame fallback path, which is what makes the
        guarantee hold even on a clip with genuinely black pixels.
    """
    with config_path.open(encoding="utf-8") as config_file:
        config: dict[str, Any] = yaml.safe_load(config_file)
    committed = config.get("analysis_region") or {}
    if region == REGION_NO_BOTTOM_EXCLUSION:
        config["analysis_region"] = {**committed, "exclude_bottom_fraction": 0.0}
    elif region == REGION_FULL_FRAME:
        config["analysis_region"] = {
            "border_threshold": 0.0,
            "exclude_bottom_fraction": 0.0,
            "min_valid_fraction": 1.0,
        }
    elif region != REGION_MASKED:
        raise ValueError(f"unknown region policy: {region}")
    return config


def _apply_weight_override(config: dict[str, Any], weights: str | None) -> dict[str, Any]:
    """Replace the fitted weights with a comma-separated triple, for comparison runs."""
    if not weights:
        return config
    parts = [part.strip() for part in weights.split(",")]
    if len(parts) != 3:
        raise ValueError("--weights takes three comma-separated values: illuminant,rolling,awb")
    config["aggregation"]["weights"] = {
        "IlluminantDetector": float(parts[0]),
        "RollingBandDetector": float(parts[1]),
        "AWBDetector": float(parts[2]),
    }
    return config


def _parse_resize(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("sampling.resize must be a two-item [width, height] list")
    return int(value[0]), int(value[1])


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score every BurstFlicker-G ground-truth frame with the calibrated detector",
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="BurstFlicker-G root holding <split>/gt/*.mp4",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to score (default: train test)",
    )
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, help="Score only the first N clips")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Parallel worker processes (default: min(cpu_count, 8); 1 runs in-process)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep clips already scored in the output and rescore only the rest",
    )
    parser.add_argument(
        "--region",
        choices=REGION_POLICIES,
        default=REGION_MASKED,
        help=(
            "Which pixels to measure over: 'masked' uses the committed fisheye "
            "policy, 'no-bottom-exclusion' keeps border detection but not the "
            "torso exclusion, 'full-frame' measures every pixel "
            f"(default: {REGION_MASKED})"
        ),
    )
    parser.add_argument(
        "--weights",
        help="Override the fitted weights as illuminant,rolling,awb (e.g. 0.10,0.65,0.25)",
    )
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if arguments.limit is not None and arguments.limit <= 0:
        parser.error("--limit must be greater than zero")

    config = _apply_weight_override(
        load_config(Path(arguments.config), region=arguments.region),
        arguments.weights,
    )
    clips = discover_clips(Path(arguments.dataset_root), arguments.splits)
    if arguments.limit is not None:
        clips = clips[: arguments.limit]

    weights = config["aggregation"]["weights"]
    logger.info(
        "Scoring %d ground-truth clips with weights illuminant=%.2f rolling=%.2f awb=%.2f",
        len(clips),
        weights.get("IlluminantDetector", 0.0),
        weights.get("RollingBandDetector", 0.0),
        weights.get("AWBDetector", 0.0),
    )
    run(
        clips,
        Path(arguments.output),
        config,
        workers=safe_worker_count(max(arguments.workers, 1), clips[0].path),
        resume=arguments.resume,
    )


if __name__ == "__main__":
    main()
