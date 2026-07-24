"""CLI for single-video and manifest-batch flicker detection."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from src.calibration.evaluation import calibrate, write_calibration_report
from src.calibration.thresholds import RollingBandNormalizer
from src.core.aggregator import DetectionAggregator, WindowMetrics
from src.core.pipeline import DetectionPipeline
from src.core.types import ManifestEntry, VideoResult
from src.data.manifest import ManifestReader
from src.data.reader import VideoReader
from src.data.sampler import WindowSampler, WindowSpec
from src.detectors.awb import AWBDetector
from src.detectors.illuminant import IlluminantDetector
from src.detectors.rolling_band import RollingBandDetector
from src.features.extractor import FeatureExtractor
from src.signals.extractor import SignalExtractor
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
OUTPUT_FIELDS = (
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
    "processing_time_seconds",
    "detector_version",
)


def process_video(
    video_path: str | Path,
    config: dict[str, Any],
    *,
    video_key: str | None = None,
) -> VideoResult:
    """Process one local path or FFmpeg-supported URL into a stable result."""
    sampler_config = config["sampling"]
    decision_config = config["decision"]
    detector_pipeline, aggregator = build_detection_components(config)
    sampler = WindowSampler(
        window_duration=sampler_config["window_duration"],
        stride=sampler_config["stride"],
    )
    resize = _parse_resize(sampler_config.get("resize"))

    started_at = perf_counter()
    window_metrics: list[WindowMetrics] = []
    window_specs: list[WindowSpec] = []
    logger.info("Processing video: %s", video_path)
    with VideoReader(video_path) as reader:
        metadata = reader.metadata()
        logger.info(
            "Video metadata: duration=%.2fs fps=%.3f dimensions=%dx%d",
            metadata.duration,
            metadata.fps,
            metadata.width,
            metadata.height,
        )
        for spec in sampler.sample(metadata.duration):
            logger.debug("Processing window start=%.2fs end=%.2fs", spec.start_time, spec.end_time)
            window = reader.read_window(
                start=spec.start_time,
                duration=spec.end_time - spec.start_time,
                resize=resize,
            )
            signals = SignalExtractor.extract(window)
            features = FeatureExtractor.extract(signals, window.fps)
            window_specs.append(spec)
            window_metrics.append(aggregator.aggregate(detector_pipeline.run(features)))

    video_metrics = aggregator.aggregate_video(window_metrics)
    worst_index = max(range(len(window_metrics)), key=lambda index: window_metrics[index].score)
    severity_band, route, confidence = classify_score(
        video_metrics.max_score,
        mild_threshold=decision_config["mild_threshold"],
        extreme_threshold=decision_config["extreme_threshold"],
    )
    result = VideoResult(
        video_key=video_key or Path(str(video_path)).name,
        flicker_score=video_metrics.max_score,
        has_flicker=severity_band != "none",
        severity_band=severity_band,
        route=route,
        confidence=confidence,
        worst_segment=(window_specs[worst_index].start_time, window_specs[worst_index].end_time),
        detector_scores=aggregator.detector_max_scores(window_metrics),
        processing_time=perf_counter() - started_at,
        detector_version=config["detector_version"],
    )
    logger.info(
        "Completed video: score=%.3f band=%s processing_time=%.2fs",
        result.flicker_score,
        result.severity_band,
        result.processing_time,
    )
    return result


def build_detection_components(
    config: dict[str, Any],
) -> tuple[DetectionPipeline, DetectionAggregator]:
    """Construct enabled detectors and their configured aggregation policy."""
    illuminant_config = config["illuminant_detector"]
    rolling_band_config = config["rolling_band_detector"]
    awb_config = config.get("awb_detector", {"enabled": False})
    aggregation_config = config["aggregation"]
    detectors = []
    if illuminant_config["enabled"]:
        detectors.append(
            IlluminantDetector(
                min_prominence=illuminant_config["peak_prominence"],
                min_ratio=illuminant_config["peak_to_median_ratio"],
            )
        )
    if rolling_band_config["enabled"]:
        detectors.append(
            RollingBandDetector(smoothing_sigma=rolling_band_config["smoothing_sigma"])
        )
    if awb_config.get("enabled", False):
        detectors.append(
            AWBDetector(
                min_chroma_std=awb_config.get("min_chroma_std", 0.5),
                ae_max_frequency=awb_config.get("ae_max_frequency", 5.0),
            )
        )

    rolling_normalizer = RollingBandNormalizer(**aggregation_config["rolling_band_normalization"])
    normalizers: dict[str, object] = {
        "IlluminantDetector": lambda result: result.score,
        "RollingBandDetector": rolling_normalizer,
    }
    if awb_config.get("enabled", False):
        normalizers["AWBDetector"] = lambda result: result.score
    aggregator = DetectionAggregator(
        weights=aggregation_config["weights"],
        normalizers=normalizers,
        positive_threshold=aggregation_config["positive_threshold"],
    )
    return DetectionPipeline(detectors), aggregator


def run_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
) -> None:
    """Process a manifest in parallel, retaining failures as stable output rows."""
    manifest = ManifestReader(manifest_path)
    output_path = Path(output_path)
    completed_keys = _completed_keys(output_path) if resume else set()
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    # Collect entries to process (filter already-completed keys)
    entries = [e for e in manifest if e.key not in completed_keys]
    if limit is not None:
        entries = entries[:limit]

    if not entries:
        logger.info("No entries to process")
        return

    n_workers = workers or min(os.cpu_count() or 1, 8)
    logger.info(
        "Processing %d videos with %d parallel workers",
        len(entries),
        n_workers,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    should_write_header = mode == "w" or (output_path.exists() and output_path.stat().st_size == 0)

    started_at = perf_counter()
    processed = 0
    failed = 0

    with output_path.open(mode, newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        if should_write_header:
            writer.writeheader()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_key = {
                executor.submit(_process_manifest_entry, entry, config): entry.key
                for entry in entries
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    row = future.result()
                except Exception as error:
                    logger.exception("Worker exception for %s", key)
                    row = _error_row(key, error, config)
                writer.writerow(row)
                output_file.flush()
                processed += 1
                if row.get("status") == "error":
                    failed += 1
                if processed % 10 == 0:
                    logger.info("Progress: %d/%d completed", processed, len(entries))

    elapsed = perf_counter() - started_at
    logger.info(
        "Finished manifest batch: processed=%d failed=%d elapsed=%.1fs output=%s",
        processed,
        failed,
        elapsed,
        output_path,
    )
    _print_throughput_projection(processed, elapsed)


def write_result_csv(result: VideoResult, output_path: str | Path) -> None:
    """Write a stable one-row flag manifest for single-video use."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerow(result_to_row(result))
    logger.info("Wrote result CSV: %s", output_path)


def result_to_row(result: VideoResult) -> dict[str, object]:
    """Convert a result to the fixed assignment flag-manifest schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "error": "",
        "video_key": result.video_key,
        "flicker_score": result.flicker_score,
        "severity_band": result.severity_band,
        "route": result.route,
        "confidence": result.confidence,
        "worst_segment_start": result.worst_segment[0],
        "worst_segment_end": result.worst_segment[1],
        "illuminant_score": result.detector_scores.get("IlluminantDetector", 0.0),
        "rolling_band_score": result.detector_scores.get("RollingBandDetector", 0.0),
        "awb_score": result.detector_scores.get("AWBDetector", 0.0),
        "processing_time_seconds": result.processing_time,
        "detector_version": result.detector_version,
    }


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


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load detector settings from YAML."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _process_manifest_entry(entry: ManifestEntry, config: dict[str, Any]) -> dict[str, object]:
    try:
        result = process_video(entry.presigned_url, config, video_key=entry.key)
        return result_to_row(result)
    except Exception as error:
        logger.exception("Failed video: %s", entry.key)
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "error": str(error),
            "video_key": entry.key,
            "flicker_score": "",
            "severity_band": "",
            "route": "",
            "confidence": "",
            "worst_segment_start": "",
            "worst_segment_end": "",
            "illuminant_score": "",
            "rolling_band_score": "",
            "awb_score": "",
            "processing_time_seconds": "",
            "detector_version": config["detector_version"],
        }


def _completed_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open(newline="", encoding="utf-8") as output_file:
        return {row["video_key"] for row in csv.DictReader(output_file) if row["video_key"]}


def _error_row(key: str, error: Exception, config: dict[str, Any]) -> dict[str, object]:
    """Build a stable error row when a worker process fails entirely."""
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "error": str(error),
        "video_key": key,
        "flicker_score": "",
        "severity_band": "",
        "route": "",
        "confidence": "",
        "worst_segment_start": "",
        "worst_segment_end": "",
        "illuminant_score": "",
        "rolling_band_score": "",
        "awb_score": "",
        "processing_time_seconds": "",
        "detector_version": config.get("detector_version", ""),
    }


def _parse_resize(value: object) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("sampling.resize must be a two-item [width, height] list")
    return int(value[0]), int(value[1])


def _print_throughput_projection(processed: int, elapsed: float) -> None:
    """Log a rough throughput projection to the full 21.7K-hour corpus."""
    if elapsed <= 0 or processed <= 0:
        return
    vps = processed / elapsed
    secs_per_video = elapsed / processed
    # Rough estimate: ~21 700 hours, average video ~5 min → ~260K videos
    corpus_hours = 21_700
    avg_video_min = 5
    corpus_videos = int(corpus_hours * 60 / avg_video_min)
    single_worker_s = corpus_videos * secs_per_video
    print(f"\n{'=' * 60}")
    print("  THROUGHPUT PROJECTION")
    print(f"{'=' * 60}")
    print(f"  Processed:      {processed} videos in {elapsed:.1f}s")
    print(f"  Rate:           {vps:.2f} videos/s ({secs_per_video:.1f}s/video)")
    print(f"\n  Full corpus (~{corpus_videos:,} videos, {corpus_hours:,} hours):")
    print(f"    1 worker:     {single_worker_s / 3600:>8.0f} hours")
    for n_workers in [8, 32, 64, 128]:
        t = single_worker_s / n_workers
        cost_cpu_h = single_worker_s / 3600
        print(
            f"    {n_workers:>3d} workers:  {t / 3600:>8.0f} hours  (~{cost_cpu_h:.0f} CPU-hours)"
        )
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect flicker in egocentric videos")
    parser.add_argument("video", nargs="?", help="One local path or FFmpeg-supported URL")
    parser.add_argument("--manifest", help="CSV manifest for batch processing")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--logging-config", default="configs/logging.yaml")
    parser.add_argument("--output", default="output/flag_manifest.csv")
    parser.add_argument("--limit", type=int, help="Maximum manifest rows to process")
    parser.add_argument(
        "--workers", type=int, help="Number of parallel workers (default: min(cpu_count, 8))"
    )
    parser.add_argument("--resume", action="store_true", help="Skip keys already in output")
    parser.add_argument("--calibrate-labels", help="CSV with video_key,label reference labels")
    parser.add_argument("--calibration-report", default="reports/calibration_report.md")
    arguments = parser.parse_args()
    input_count = int(bool(arguments.video)) + int(bool(arguments.manifest))
    if arguments.calibrate_labels:
        if input_count:
            parser.error("calibration does not accept VIDEO or --manifest")
    elif input_count != 1:
        parser.error("provide exactly one of VIDEO or --manifest")

    configure_logging(arguments.logging_config)
    config = load_config(arguments.config)
    try:
        if arguments.calibrate_labels:
            result = calibrate(arguments.output, arguments.calibrate_labels)
            write_calibration_report(result, arguments.calibration_report)
            print(
                "Calibration complete: "
                f"mild={result.mild_threshold:.4f} extreme={result.extreme_threshold:.4f}"
            )
        elif arguments.manifest:
            run_manifest(
                arguments.manifest,
                arguments.output,
                config,
                limit=arguments.limit,
                resume=arguments.resume,
                workers=arguments.workers,
            )
        else:
            result = process_video(arguments.video, config)
            write_result_csv(result, arguments.output)
            print(f"{result.video_key}: flicker_score={result.flicker_score:.3f}")
    except Exception:
        logger.exception("Flicker detection failed")
        raise


if __name__ == "__main__":
    main()
