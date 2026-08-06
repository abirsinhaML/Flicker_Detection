"""CLI for single-video and manifest-batch flicker detection."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from itertools import chain, islice
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

from src.calibration.discrimination import measure_discrimination, write_discrimination_report
from src.calibration.evaluation import calibrate, write_calibration_report
from src.calibration.thresholds import (
    AWBNormalizer,
    IlluminantNormalizer,
    RollingBandNormalizer,
)
from src.core.aggregator import DetectionAggregator, ScoreNormalizer, WindowMetrics
from src.core.pipeline import DetectionPipeline
from src.core.types import ManifestEntry, VideoResult
from src.data.decode import BACKENDS, DecodePolicy
from src.data.manifest import ManifestReader, key_for_uri, write_manifest
from src.data.reader import VideoReader
from src.data.s3_source import S3AccessError, S3Settings, SourceResolver, redact_url
from src.data.sampler import WindowSampler, WindowSpec
from src.detectors.awb import AWBDetector
from src.detectors.base import BaseDetector
from src.detectors.illuminant import IlluminantDetector
from src.detectors.rolling_band import RollingBandDetector
from src.features.extractor import FeatureExtractor
from src.signals.extractor import SignalExtractor
from src.signals.mask import RegionPolicy
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"
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
    # Diagnostics, not evidence: they qualify how far the scores can be trusted.
    "horizontal_coherence",
    "valid_fraction",
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
    decode_policy = DecodePolicy.from_config(config.get("decode"))
    region_policy = RegionPolicy.from_config(config.get("analysis_region"))
    flicker_band = config.get("flicker_band") or {}
    min_flicker = float(flicker_band.get("min_frequency") or 0.0)
    max_flicker_value = flicker_band.get("max_frequency")
    max_flicker = float(max_flicker_value) if max_flicker_value is not None else None

    started_at = perf_counter()
    window_metrics: list[WindowMetrics] = []
    window_specs: list[WindowSpec] = []
    diagnostics: list[tuple[float, float]] = []
    logger.info("Processing video: %s", video_key or redact_url(str(video_path)))
    with VideoReader(video_path, decode_policy=decode_policy, resize=resize) as reader:
        metadata = reader.metadata()
        logger.info(
            "Video metadata: duration=%.2fs fps=%.3f dimensions=%dx%d decode=%s",
            metadata.duration,
            metadata.fps,
            metadata.width,
            metadata.height,
            reader.decode_backend,
        )
        for spec in sampler.sample(metadata.duration):
            logger.debug("Processing window start=%.2fs end=%.2fs", spec.start_time, spec.end_time)
            # The reader already knows the target size, and applies it inside the
            # decoder where the backend allows, so it is not repeated per window.
            window = reader.read_window(
                start=spec.start_time,
                duration=spec.end_time - spec.start_time,
            )
            signals = SignalExtractor.extract(window, region_policy)
            features = FeatureExtractor.extract(
                signals,
                window.fps,
                min_flicker_frequency=min_flicker,
                max_flicker_frequency=max_flicker,
            )
            window_specs.append(spec)
            results = detector_pipeline.run(features)
            window_metrics.append(aggregator.aggregate(results))
            diagnostics.append(
                (_horizontal_coherence(results), features.valid_fraction),
            )

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
        # Scores from the worst window, not per-detector maxima across windows.
        # Taking each detector's maximum independently mixed evidence from
        # different moments, so the columns did not sum to flicker_score and
        # could not explain the routing decision they accompanied.
        detector_scores=dict(window_metrics[worst_index].detector_scores),
        processing_time=perf_counter() - started_at,
        detector_version=config["detector_version"],
        horizontal_coherence=diagnostics[worst_index][0],
        valid_fraction=diagnostics[worst_index][1],
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

    # Detectors measure; normalizers decide what a measurement is worth.  Every
    # tunable scale therefore lives in the aggregation config, where it can be
    # fitted against reference labels.
    detectors: list[BaseDetector] = []
    normalizers: dict[str, ScoreNormalizer] = {}
    if illuminant_config["enabled"]:
        detectors.append(IlluminantDetector())
        normalizers["IlluminantDetector"] = IlluminantNormalizer(
            **aggregation_config["illuminant_normalization"]
        )
    if rolling_band_config["enabled"]:
        detectors.append(
            RollingBandDetector(smoothing_sigma=rolling_band_config["smoothing_sigma"])
        )
        normalizers["RollingBandDetector"] = RollingBandNormalizer(
            **aggregation_config["rolling_band_normalization"]
        )
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


def run_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
) -> None:
    """Process every video named by a manifest file."""
    settings = S3Settings.from_config(config)
    manifest = ManifestReader(manifest_path, default_bucket=settings.bucket)
    logger.info("Read %d entries from manifest: %s", len(manifest), manifest_path)
    run_batch(
        manifest,
        output_path,
        config,
        limit=limit,
        resume=resume,
        workers=workers,
    )


def run_s3_prefix(
    listing_root: str | None,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
) -> None:
    """List an S3 prefix and process every video under it.

    The listing streams, so a prefix holding the full corpus never has to be
    materialized before work starts.
    """
    catalogs = S3Settings.from_config(config).catalogs(listing_root)
    for catalog in catalogs:
        catalog.verify_access()
        logger.info("Listing videos under s3://%s/%s", catalog.bucket, catalog.prefix)
    
    entries = chain.from_iterable(catalog.list_entries() for catalog in catalogs)
    run_batch(
        entries,
        output_path,
        config,
        limit=limit,
        resume=resume,
        workers=workers,
    )


def run_batch(
    entries: Iterable[ManifestEntry],
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
) -> None:
    """Score a stream of videos in parallel, keeping failures as output rows.

    ``entries`` is consumed lazily and only a bounded number of videos are ever
    in flight, so peak memory does not scale with corpus size.  Each row is
    flushed as it completes, making the output safe to resume after expired
    credentials or a network failure.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    output_path = Path(output_path)
    # Only successful rows count as done.  Failures are retried, because the
    # expected failures here are transient: expired credentials, throttling, or
    # a dropped connection part-way through an object.
    retained_rows = _successful_rows(output_path) if resume else {}
    if retained_rows:
        logger.info("Resuming: %d videos already scored in %s", len(retained_rows), output_path)

    # Limit selects the first N videos of the work set *before* the resume
    # filter, so `--limit N --resume` converges on the same N videos rather than
    # pulling N further ones each run.
    selected = islice(entries, limit) if limit is not None else entries
    pending: Iterator[ManifestEntry] = (
        entry for entry in selected if entry.key not in retained_rows
    )
    first_entry = next(pending, None)
    if first_entry is None:
        logger.info("No entries to process")
        return
    pending = chain([first_entry], pending)

    n_workers = workers or min(os.cpu_count() or 1, 8)
    # Keep the queue deep enough to hide per-video latency spikes without
    # holding the whole listing in memory.
    max_inflight = n_workers * 4
    resolver = S3Settings.from_config(config).resolver()
    # Fail the batch here rather than turning a credential problem into one
    # error row per video across the whole corpus.
    resolver.verify_readable(first_entry.source_uri)
    logger.info("Processing videos with %d parallel workers", n_workers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = perf_counter()
    processed = 0
    failed = 0

    # Rewriting the retained rows keeps exactly one row per video key, while
    # flushing each new row keeps a killed batch resumable.
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(retained_rows.values())
        output_file.flush()

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight: dict[Future[dict[str, object]], str] = {}
            exhausted = False
            while True:
                while not exhausted and len(inflight) < max_inflight:
                    entry = next(pending, None)
                    if entry is None:
                        exhausted = True
                        break
                    future = executor.submit(_process_manifest_entry, entry, config, resolver)
                    inflight[future] = entry.key
                if not inflight:
                    break

                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    key = inflight.pop(future)
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
                        logger.info("Progress: %d completed, %d failed", processed, failed)

    elapsed = perf_counter() - started_at
    logger.info(
        "Finished batch: processed=%d failed=%d elapsed=%.1fs output=%s",
        processed,
        failed,
        elapsed,
        output_path,
    )
    _print_throughput_projection(processed, elapsed)


def snapshot_s3_manifest(
    listing_root: str | None,
    manifest_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
) -> int:
    """Write the current S3 listing to a durable, credential-free manifest.

    Pinning the listing this way makes a run reproducible: the same manifest
    yields the same work set even as the bucket gains objects.
    """
    catalogs = S3Settings.from_config(config).catalogs(listing_root)
    for catalog in catalogs:
        catalog.verify_access()
    
    entries = chain.from_iterable(catalog.list_entries(limit=limit) for catalog in catalogs)
    written = write_manifest(entries, manifest_path)
    logger.info(
        "Wrote %d entries from %d prefixes to %s",
        written,
        len(catalogs),
        manifest_path,
    )
    return written


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
        "horizontal_coherence": result.horizontal_coherence,
        "valid_fraction": result.valid_fraction,
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


def _process_manifest_entry(
    entry: ManifestEntry,
    config: dict[str, Any],
    resolver: SourceResolver,
) -> dict[str, object]:
    """Sign, decode, and score one video inside a worker process.

    Signing happens here rather than in the parent so the URL is seconds old
    when FFmpeg opens it, and so no signed URL is ever written to disk or held
    for the lifetime of a long batch.
    """
    try:
        source = resolver.resolve(entry.source_uri)
        result = process_video(source, config, video_key=entry.key)
        return result_to_row(result)
    except Exception as error:
        logger.exception("Failed video: %s", entry.key)
        return _error_row(entry.key, error, config)


def _successful_rows(output_path: Path) -> dict[str, dict[str, object]]:
    """Return the successful rows of an existing flag manifest, keyed by video.

    Unknown columns are dropped and duplicate keys collapse to the last row, so
    an output written by an earlier detector version still resumes cleanly.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return {}
    rows: dict[str, dict[str, object]] = {}
    with output_path.open(newline="", encoding="utf-8") as output_file:
        for row in csv.DictReader(output_file):
            key = (row.get("video_key") or "").strip()
            if not key or row.get("status") != "ok":
                continue
            rows[key] = {field: row.get(field, "") for field in OUTPUT_FIELDS}
    return rows


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


def _horizontal_coherence(results: dict[str, object]) -> float:
    """Pull the band-model self-check out of the rolling-band measurements."""
    metrics = results.get("RollingBandDetector")
    return float(getattr(metrics, "horizontal_coherence", 1.0))


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
    parser.add_argument(
        "video",
        nargs="?",
        help="One video: an s3://bucket/key URI, local path, or FFmpeg-supported URL",
    )
    parser.add_argument("--manifest", help="Manifest of videos to score (CSV or .txt list)")
    parser.add_argument(
        "--s3-prefix",
        nargs="?",
        const="",
        help=(
            "Score every video under an S3 prefix, listed live. Accepts a full "
            "s3://bucket/prefix URI, a bare prefix read from the configured "
            "bucket, or no value to use the configured bucket and prefix."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        help=(
            "Snapshot the --s3-prefix listing to this durable manifest and exit "
            "without scoring. Pin a run by scoring the snapshot with --manifest."
        ),
    )
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--logging-config", default="configs/logging.yaml")
    parser.add_argument("--output", default="output/flag_manifest.csv")
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N videos of the work set (stable across --resume)",
    )
    parser.add_argument(
        "--workers", type=int, help="Number of parallel workers (default: min(cpu_count, 8))"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep videos already scored in the output and retry only the rest",
    )
    parser.add_argument(
        "--decode-backend",
        choices=BACKENDS,
        help="Override decode.backend: auto (NVDEC when usable), cuda, or cpu",
    )
    parser.add_argument("--aws-profile", help="AWS named profile (default: standard chain)")
    parser.add_argument("--aws-region", help="Bucket region for SigV4 signing")
    parser.add_argument("--calibrate-labels", help="CSV with video_key,label reference labels")
    parser.add_argument("--calibration-report", default="reports/calibration_report.md")
    parser.add_argument("--discrimination-report", default="reports/discrimination_report.md")
    arguments = parser.parse_args()
    listing_requested = arguments.s3_prefix is not None
    input_count = (
        int(bool(arguments.video)) + int(bool(arguments.manifest)) + int(listing_requested)
    )
    if arguments.write_manifest and not listing_requested:
        parser.error("--write-manifest requires --s3-prefix")
    if arguments.calibrate_labels:
        if input_count:
            parser.error("calibration does not accept VIDEO, --manifest, or --s3-prefix")
    elif input_count != 1:
        parser.error("provide exactly one of VIDEO, --manifest, or --s3-prefix")

    configure_logging(arguments.logging_config)
    config = _apply_decode_overrides(
        _apply_aws_overrides(load_config(arguments.config), arguments), arguments
    )
    try:
        if arguments.calibrate_labels:
            # Rank every signal before fitting anything. A threshold fit on an
            # inverted score still reports a number, so the ordering check has to
            # come first and be visible next to the fitted bands.
            discrimination = measure_discrimination(arguments.output, arguments.calibrate_labels)
            write_discrimination_report(discrimination, arguments.discrimination_report)
            for measured in discrimination:
                print(
                    f"  {measured.signal:>22} spearman={measured.spearman:+.3f} "
                    f"AUC={measured.auc_extreme_vs_none:.3f}"
                    f"{'   INVERTED' if measured.is_inverted else ''}"
                )
            result = calibrate(arguments.output, arguments.calibrate_labels)
            write_calibration_report(result, arguments.calibration_report)
            print(
                "Calibration complete: "
                f"mild={result.mild_threshold:.4f} extreme={result.extreme_threshold:.4f}"
            )
        elif arguments.write_manifest:
            written = snapshot_s3_manifest(
                arguments.s3_prefix,
                arguments.write_manifest,
                config,
                limit=arguments.limit,
            )
            print(f"Wrote {written} entries to {arguments.write_manifest}")
        elif listing_requested:
            run_s3_prefix(
                arguments.s3_prefix,
                arguments.output,
                config,
                limit=arguments.limit,
                resume=arguments.resume,
                workers=arguments.workers,
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
            result = process_one(arguments.video, config)
            write_result_csv(result, arguments.output)
            print(f"{result.video_key}: flicker_score={result.flicker_score:.3f}")
    except S3AccessError as error:
        logger.error("S3 access failed: %s", error)
        raise SystemExit(2) from error
    except Exception:
        logger.exception("Flicker detection failed")
        raise


def process_one(source_uri: str, config: dict[str, Any]) -> VideoResult:
    """Score a single video named by an S3 URI, local path, or URL."""
    resolver = S3Settings.from_config(config).resolver()
    return process_video(
        resolver.resolve(source_uri),
        config,
        video_key=key_for_uri(source_uri),
    )


def _apply_aws_overrides(config: dict[str, Any], arguments: argparse.Namespace) -> dict[str, Any]:
    """Let CLI flags override the config's ``s3`` block."""
    overrides = {"profile": arguments.aws_profile, "region": arguments.aws_region}
    supplied = {name: value for name, value in overrides.items() if value}
    if supplied:
        config["s3"] = {**(config.get("s3") or {}), **supplied}
    return config


def _apply_decode_overrides(
    config: dict[str, Any],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    """Let ``--decode-backend`` override the config's ``decode`` block."""
    if arguments.decode_backend:
        config["decode"] = {**(config.get("decode") or {}), "backend": arguments.decode_backend}
    return config


if __name__ == "__main__":
    main()
