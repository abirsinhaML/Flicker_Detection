"""CLI for single-video and manifest-batch flicker detection."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime, timezone
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
from src.core.aggregator import DetectionAggregator, ScoreNormalizer
from src.core.pipeline import DetectionPipeline
from src.core.records import (
    SCHEMA_VERSION,
    VIDEO_FIELDS,
    WINDOW_FIELDS,
    error_record,
    measurements_of,
    result_to_record,
)
from src.core.types import ManifestEntry, VideoResult, WindowRecord
from src.data.decode import BACKENDS, DecodePolicy
from src.data.links import (
    BucketKeyIndex,
    LinkResolution,
    LinkSheetReader,
    resolve_links,
    write_resolution_report,
)
from src.data.manifest import ManifestReader, key_for_uri, write_manifest
from src.data.publish import S3Publisher
from src.data.reader import VideoReader
from src.data.results import (
    ResultWriter,
    read_successful_records,
)
from src.data.rows import shard_ranges, validate_row_range, weighted_shard_ranges
from src.data.s3_source import (
    S3AccessError,
    S3Settings,
    SourceResolver,
    redact_text,
    redact_url,
)
from src.data.sampler import WindowSampler
from src.detectors.awb import AWBDetector
from src.detectors.base import BaseDetector
from src.detectors.illuminant import IlluminantDetector
from src.detectors.rolling_band import RollingBandDetector
from src.features.extractor import FeatureExtractor
from src.signals.extractor import SignalExtractor
from src.signals.mask import RegionPolicy
from src.utils.logger import configure_logging

logger = logging.getLogger(__name__)

__all__ = [
    "SCHEMA_VERSION",
    "VIDEO_FIELDS",
    "WINDOW_FIELDS",
    "classify_score",
    "load_config",
    "process_video",
    "run_manifest",
    "write_result",
]


def process_video(
    video_path: str | Path,
    config: dict[str, Any],
    *,
    video_key: str | None = None,
    entry: ManifestEntry | None = None,
) -> VideoResult:
    """Process one local path or FFmpeg-supported URL into a stable result.

    Every sampled window is scored, graded, and retained.  The video-level
    rollup is then derived from those windows rather than being the only thing
    computed, so the reported maximum can always be located in the timeline and
    compared against the windows it beat.
    """
    sampler_config = config["sampling"]
    decision_config = config["decision"]
    window_mild, window_extreme = window_thresholds(decision_config)
    detector_pipeline, aggregator = build_detection_components(config)
    sampler = WindowSampler(
        window_duration=sampler_config["window_duration"],
        # Absent or null means contiguous, so a config that only sets the window
        # length tiles the video rather than silently leaving gaps in it.
        stride=sampler_config.get("stride"),
    )
    resize = _parse_resize(sampler_config.get("resize"))
    decode_policy = DecodePolicy.from_config(config.get("decode"))
    region_policy = RegionPolicy.from_config(config.get("analysis_region"))
    flicker_band = config.get("flicker_band") or {}
    min_flicker = float(flicker_band.get("min_frequency") or 0.0)
    max_flicker_value = flicker_band.get("max_frequency")
    max_flicker = float(max_flicker_value) if max_flicker_value is not None else None

    started_at = perf_counter()
    windows: list[WindowRecord] = []
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
        decode_backend = reader.decode_backend
        # The whole schedule is handed over at once so the reader can decode a
        # gapless one in a single forward pass instead of seeking per window.
        # Windows arrive as they complete, so only one is held at a time.
        schedule = list(sampler.sample(metadata.duration))
        for index, window in enumerate(reader.read_windows(schedule)):
            logger.debug(
                "Processing window start=%.2fs end=%.2fs", window.start_time, window.end_time
            )
            signals = SignalExtractor.extract(window, region_policy)
            features = FeatureExtractor.extract(
                signals,
                window.fps,
                min_flicker_frequency=min_flicker,
                max_flicker_frequency=max_flicker,
            )
            results = detector_pipeline.run(features)
            metrics = aggregator.aggregate(results)
            band, route, confidence = classify_score(
                metrics.score,
                mild_threshold=window_mild,
                extreme_threshold=window_extreme,
            )
            windows.append(
                WindowRecord(
                    index=index,
                    start_time=window.start_time,
                    end_time=window.end_time,
                    fps=window.fps,
                    frame_count=int(window.frames.shape[0]),
                    score=metrics.score,
                    severity_band=band,
                    route=route,
                    confidence=confidence,
                    detector_scores=dict(metrics.detector_scores),
                    measurements=measurements_of(results),
                    valid_fraction=features.valid_fraction,
                    horizontal_coherence=_horizontal_coherence(results),
                )
            )

    video_metrics = aggregator.aggregate_video(windows)
    worst_index = max(range(len(windows)), key=lambda index: windows[index].score)
    windows[worst_index].is_worst = True
    worst = windows[worst_index]
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
        worst_segment=(worst.start_time, worst.end_time),
        # Scores from the worst window, not per-detector maxima across windows.
        # Taking each detector's maximum independently mixed evidence from
        # different moments, so the columns did not sum to flicker_score and
        # could not explain the routing decision they accompanied.
        detector_scores=dict(worst.detector_scores),
        processing_time=perf_counter() - started_at,
        detector_version=config["detector_version"],
        horizontal_coherence=worst.horizontal_coherence,
        valid_fraction=worst.valid_fraction,
        windows=windows,
        mean_score=video_metrics.mean_score,
        positive_windows=video_metrics.positive_windows,
        total_windows=video_metrics.total_windows,
        worst_index=worst_index,
        duration=metadata.duration,
        video_fps=metadata.fps,
        width=metadata.width,
        height=metadata.height,
        decode_backend=decode_backend,
        project_name=(entry.project_name or "") if entry else "",
        video_id=(entry.video_id or "") if entry else "",
        sheet_duration=entry.duration_seconds if entry else None,
    )
    logger.info(
        "Completed video: windows=%d score=%.3f (mean %.3f) band=%s "
        "worst=[%.1f, %.1f]s processing_time=%.2fs",
        result.total_windows,
        result.flicker_score,
        result.mean_score,
        result.severity_band,
        result.worst_segment[0],
        result.worst_segment[1],
        result.processing_time,
    )
    return result


def window_thresholds(decision_config: dict[str, Any]) -> tuple[float, float]:
    """Return the per-window severity boundaries, defaulting to the video ones.

    They are a *separate* calibration, and deliberately configurable apart from
    the video thresholds, because the two quantities are not drawn from the same
    distribution.  A video score is a maximum over ~31 windows, which sits far up
    the per-window distribution: bootstrapping the clean ground-truth clips puts
    the median of max-of-15 at 0.0164 against 0.0005 for a single window.  A
    boundary fitted against video maxima is therefore too high to apply to an
    individual window, and window bands under it read as conservative.

    Defaulting to the video thresholds keeps one documented number rather than
    inventing a second one; ``decision.window`` is where a fitted value goes once
    window-level labels exist.
    """
    overrides = decision_config.get("window") or {}
    mild = float(overrides.get("mild_threshold", decision_config["mild_threshold"]))
    extreme = float(overrides.get("extreme_threshold", decision_config["extreme_threshold"]))
    return mild, extreme


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
    video_csv: str | Path | None = None,
    window_dir: str | Path | None = None,
    publisher: S3Publisher | None = None,
    publish_root: str | Path | None = None,
    from_row: int | None = None,
    to_row: int | None = None,
) -> None:
    """Process every video named by a manifest file."""
    settings = S3Settings.from_config(config)
    manifest = ManifestReader(
        manifest_path,
        default_bucket=settings.bucket,
        from_row=from_row,
        to_row=to_row,
    )
    logger.info("Manifest %s: %s", manifest_path, manifest.row_range)
    run_batch(
        manifest,
        output_path,
        config,
        limit=limit,
        resume=resume,
        workers=workers,
        video_csv=video_csv,
        window_dir=window_dir,
        publisher=publisher,
        publish_root=publish_root,
    )


def link_entries(
    sheet_path: str | Path,
    config: dict[str, Any],
    *,
    resolve_case: bool = True,
    report_path: str | Path | None = None,
    from_row: int | None = None,
    to_row: int | None = None,
) -> Iterator[ManifestEntry]:
    """Yield the work set named by a link sheet's ``video_s3_link`` column.

    The key comes from the column and the bucket from ``s3.bucket``.  The sheet's
    links are lowercased while S3 keys are case-sensitive, so unless
    ``resolve_case`` is off the bucket is listed once to recover the real casing;
    without it every row 404s.  See :mod:`src.data.links`.

    The report is written *before* scoring starts, so the rows that will not be
    attempted are on disk even if the batch is later killed.
    """
    settings = S3Settings.from_config(config)
    if not settings.bucket:
        raise ValueError("s3.bucket must be set to resolve link-sheet keys")

    sheet = LinkSheetReader(sheet_path, from_row=from_row, to_row=to_row)
    logger.info(
        "Link sheet %s (column %r): %s", sheet_path, sheet.link_column, sheet.row_range
    )
    index = None
    if resolve_case:
        index = BucketKeyIndex.from_catalog(settings.catalog())

    resolution = LinkResolution()
    entries = list(resolve_links(sheet, settings.bucket, index=index, resolution=resolution))
    logger.info("Link resolution: %s", resolution.summary())
    if resolution.skipped:
        logger.warning(
            "Skipping %d unresolvable links (%d missing, %d ambiguous, %d duplicate)",
            resolution.skipped,
            len(resolution.missing),
            len(resolution.ambiguous),
            len(resolution.duplicates),
        )
        if report_path:
            write_resolution_report(resolution, report_path, sheet_path=sheet_path)
    yield from entries


def run_links(
    sheet_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
    video_csv: str | Path | None = None,
    window_dir: str | Path | None = None,
    publisher: S3Publisher | None = None,
    publish_root: str | Path | None = None,
    resolve_case: bool = True,
    report_path: str | Path | None = None,
    from_row: int | None = None,
    to_row: int | None = None,
) -> None:
    """Score every video named by a link sheet."""
    run_batch(
        link_entries(
            sheet_path,
            config,
            resolve_case=resolve_case,
            report_path=report_path,
            from_row=from_row,
            to_row=to_row,
        ),
        output_path,
        config,
        limit=limit,
        resume=resume,
        workers=workers,
        video_csv=video_csv,
        window_dir=window_dir,
        publisher=publisher,
        publish_root=publish_root,
    )


def run_s3_prefix(
    listing_root: str | None,
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
    video_csv: str | Path | None = None,
    window_dir: str | Path | None = None,
    publisher: S3Publisher | None = None,
    publish_root: str | Path | None = None,
) -> None:
    """List an S3 prefix and process every video under it.

    The listing streams, so a prefix holding the full corpus never has to be
    materialized before work starts.
    """
    catalog = S3Settings.from_config(config).catalog(listing_root)
    catalog.verify_access()
    logger.info("Listing videos under s3://%s/%s", catalog.bucket, catalog.prefix)
    run_batch(
        catalog.list_entries(),
        output_path,
        config,
        limit=limit,
        resume=resume,
        workers=workers,
        video_csv=video_csv,
        window_dir=window_dir,
        publisher=publisher,
        publish_root=publish_root,
    )


def run_batch(
    entries: Iterable[ManifestEntry],
    output_path: str | Path,
    config: dict[str, Any],
    *,
    limit: int | None = None,
    resume: bool = False,
    workers: int | None = None,
    video_csv: str | Path | None = None,
    window_dir: str | Path | None = None,
    publisher: S3Publisher | None = None,
    publish_root: str | Path | None = None,
) -> None:
    """Score a stream of videos in parallel, keeping failures as output records.

    ``entries`` is consumed lazily and only a bounded number of videos are ever
    in flight, so peak memory does not scale with corpus size.  Each video's
    record is flushed as it completes, making the output safe to resume after
    expired credentials or a network failure.

    A video is one JSONL line however many windows it contains, so the unit that
    is appended and the unit that ``--resume`` skips remain the same thing.  The
    flat CSVs are derived as records land, never a separate pass.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    output_path = Path(output_path)
    # Only successful records count as done.  Failures are retried, because the
    # expected failures here are transient: expired credentials, throttling, or
    # a dropped connection part-way through an object.
    retained = read_successful_records(output_path) if resume else {}
    if retained:
        logger.info("Resuming: %d videos already scored in %s", len(retained), output_path)

    # Limit selects the first N videos of the work set *before* the resume
    # filter, so `--limit N --resume` converges on the same N videos rather than
    # pulling N further ones each run.
    selected = islice(entries, limit) if limit is not None else entries
    pending: Iterator[ManifestEntry] = (entry for entry in selected if entry.key not in retained)
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
    # error record per video across the whole corpus.
    resolver.verify_readable(first_entry.source_uri)
    logger.info("Processing videos with %d parallel workers", n_workers)

    started_at = perf_counter()
    processed = 0
    failed = 0
    windows_written = 0

    # Rewriting the retained records keeps exactly one record per video key,
    # while flushing each new one keeps a killed batch resumable.
    writer = ResultWriter(
        output_path,
        video_csv_path=video_csv,
        window_dir=window_dir,
        publisher=publisher,
        publish_root=publish_root,
    )
    with writer:
        writer.open(retained.values())

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            inflight: dict[Future[dict[str, Any]], str] = {}
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
                        record = future.result()
                    except Exception as error:
                        logger.exception("Worker exception for %s", key)
                        record = _error_record(key, error, config)
                    writer.write(record)
                    processed += 1
                    if record.get("status") == "error":
                        failed += 1
                    else:
                        windows_written += len(record.get("windows") or ())
                    if processed % 10 == 0:
                        logger.info(
                            "Progress: %d completed, %d failed, %d windows",
                            processed,
                            failed,
                            windows_written,
                        )

    if publisher is not None:
        # After the loop, so the JSONL and rollup are complete rather than
        # a snapshot of whatever had landed when the last video finished.
        writer.publish_summary()
        logger.info(
            "Published %d files to %s (%d failed)",
            publisher.uploaded,
            publisher.uri,
            publisher.failed,
        )

    elapsed = perf_counter() - started_at
    logger.info(
        "Finished batch: processed=%d failed=%d windows=%d elapsed=%.1fs output=%s",
        processed,
        failed,
        windows_written,
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
    catalog = S3Settings.from_config(config).catalog(listing_root)
    catalog.verify_access()
    written = write_manifest(catalog.list_entries(limit=limit), manifest_path)
    logger.info("Wrote %d entries to %s", written, manifest_path)
    return written


def write_result(
    result: VideoResult,
    output_path: str | Path,
    *,
    video_csv: str | Path | None = None,
    window_dir: str | Path | None = None,
    publisher: S3Publisher | None = None,
    publish_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write one video's record, and any requested flat tables, then return it."""
    record = result_to_record(result, completed_at=_timestamp())
    writer = ResultWriter(
        output_path,
        video_csv_path=video_csv,
        window_dir=window_dir,
        publisher=publisher,
        publish_root=publish_root,
    )
    with writer:
        writer.open()
        writer.write(record)
    # After close, so the records file is flushed before it is copied up.  Without
    # this a single-video run with --s3-output uploaded only its window CSV and
    # silently left the record and rollup behind.
    writer.publish_summary()
    logger.info(
        "Wrote %d window records for %s to %s",
        len(record["windows"]),
        result.video_key,
        output_path,
    )
    return record


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
) -> dict[str, Any]:
    """Sign, decode, and score one video inside a worker process.

    Signing happens here rather than in the parent so the URL is seconds old
    when FFmpeg opens it, and so no signed URL is ever written to disk or held
    for the lifetime of a long batch.
    """
    try:
        source = resolver.resolve(entry.source_uri)
        result = process_video(source, config, video_key=entry.key, entry=entry)
        return result_to_record(result, completed_at=_timestamp())
    except Exception as error:
        logger.exception("Failed video: %s", entry.key)
        return _error_record(entry.key, error, config)


def _error_record(key: str, error: Exception, config: dict[str, Any]) -> dict[str, Any]:
    """Build a stable error record when a worker process fails entirely.

    The message is redacted before it is stored.  FFmpeg reports a failed open by
    quoting the whole URL back, so without this a presigned URL's signature --
    a bearer credential for the object -- is written verbatim into the output and
    outlives the run that produced it.
    """
    return error_record(
        key,
        redact_text(str(error)),
        config.get("detector_version", ""),
        completed_at=_timestamp(),
    )


def _timestamp() -> str:
    """When this record was produced, in UTC.

    Written in the worker as the record is built, so it marks the moment a video
    finished rather than when the parent got round to flushing it.  UTC keeps
    the column sortable and immune to the host's timezone.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    """Log a rough throughput projection to the full corpus.

    The corpus figures are the link sheet's own: 140,335 resolvable videos
    totalling 28,216 hours, which replaces the earlier guess of 260K videos at an
    assumed five-minute average.  Projecting per *video* still assumes this
    sample's videos are of typical length, and the sheet's durations range from
    0.02 s to 3.7 hours, so read the projection as an order of magnitude.
    """
    if elapsed <= 0 or processed <= 0:
        return
    vps = processed / elapsed
    secs_per_video = elapsed / processed
    corpus_videos = 140_335
    corpus_hours = 28_216
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
    # The rows above divide by the worker count, which only holds while workers
    # are the constraint.  On the hardware path they are not: an A10G's decode
    # engine measured 100% busy with its SMs at 8%, and five workers bought 2.5x
    # a single one rather than 5x.  Treat the table as a floor on time, not a
    # forecast, and measure the knee on the machine you are actually using.
    print("\n  NOTE: assumes linear scaling in --workers. With NVDEC the decode")
    print("  engine saturates first (measured: ~2.5x from 5 workers on one A10G),")
    print("  so the rows above are optimistic beyond a handful of workers.")
    print(f"{'=' * 60}\n")


def _print_window_summary(result: VideoResult) -> None:
    """Print the per-window table for one video.

    The video line alone cannot say whether a score came from one bad moment or a
    continuously affected clip, which is the first thing a reviewer needs.  The
    table is printed in time order, with the window that set the video's score
    marked, so the two are visibly the same number.
    """
    counts = {"none": 0, "mild": 0, "extreme": 0}
    for window in result.windows:
        counts[window.severity_band] = counts.get(window.severity_band, 0) + 1

    print(f"\n{result.video_key}")
    print(
        f"  {result.duration:.1f}s at {result.video_fps:.2f} fps, "
        f"{result.width}x{result.height}, decode={result.decode_backend}"
    )
    print(
        f"  flicker_score={result.flicker_score:.4f} (max of {result.total_windows} windows, "
        f"mean {result.mean_score:.4f}) band={result.severity_band} route={result.route}"
    )
    print(
        f"  windows: {counts['none']} none, {counts['mild']} mild, "
        f"{counts['extreme']} extreme, {result.positive_windows} above positive_threshold"
    )
    header = (
        f"\n  {'#':>3}  {'start':>8}  {'end':>8}  {'score':>7}  {'band':<7}  "
        f"{'illum':>6}  {'band_r':>6}  {'awb':>6}  {'freq':>6}  {'valid':>6}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 4)}")
    for window in result.windows:
        measured = window.measurements.get("illuminant") or {}
        print(
            f"  {window.index:>3}  {window.start_time:>8.2f}  {window.end_time:>8.2f}  "
            f"{window.score:>7.4f}  {window.severity_band:<7}  "
            f"{window.detector_scores.get('IlluminantDetector', 0.0):>6.3f}  "
            f"{window.detector_scores.get('RollingBandDetector', 0.0):>6.3f}  "
            f"{window.detector_scores.get('AWBDetector', 0.0):>6.3f}  "
            f"{float(measured.get('dominant_frequency', 0.0)):>6.2f}  "
            f"{window.valid_fraction:>6.3f}"
            f"{'   <- worst' if window.is_worst else ''}"
        )
    print()


def print_shards(arguments: argparse.Namespace, config: dict[str, Any]) -> None:
    """Print the row ranges that tile the input into N shards.

    The shard count is whatever the operator asks for: how many instances are
    available is their decision, and hard-coding a number into the repo only means
    the two disagree later.  Counting the rows here rather than leaving it to them
    is the point -- the ranges are derived from the file that will actually be
    read, so they cannot drift from it.

    ``--shard-format tsv`` emits ``index<TAB>from<TAB>to`` and nothing else, so a
    launcher can read it in a loop instead of the ranges being retyped, which is
    where fleet gaps and double-scoring come from.
    """
    if arguments.links is not None:
        source = _links_path(arguments, config, None)
        reader = LinkSheetReader(source)
        frame = reader.frame
        durations = frame[reader.duration_column].tolist() if reader.duration_column else []
    elif arguments.manifest:
        source = arguments.manifest
        manifest = ManifestReader(source)
        frame = manifest.df
        durations = frame["duration_s"].tolist() if "duration_s" in frame.columns else []
    else:
        raise SystemExit("--print-shards needs --links or --manifest")

    total = len(frame)
    # Balance on duration when the input carries it: decode is ~94% of the cost,
    # so hours predict runtime and row counts do not.  Equal row counts leave the
    # heaviest of six shards with 1.20x the hours of the lightest.
    if durations and arguments.shard_by == "duration":
        ranges = weighted_shard_ranges(durations, arguments.print_shards)
        basis = "balanced by duration"
    else:
        ranges = shard_ranges(total, arguments.print_shards)
        basis = "balanced by row count"

    if arguments.shard_format == "tsv":
        for index, (first, last) in enumerate(ranges):
            print(f"{index}\t{first}\t{last}")
        return

    total_hours = sum(durations) / 3600 if durations else 0.0
    print(f"\n{total:,} rows in {source} -> {len(ranges)} shards, {basis}")
    if total_hours:
        print(f"{total_hours:,.0f} hours of video total\n")
    header = f"  {'shard':>5}  {'--from-row':>11}  {'--to-row':>11}  {'rows':>9}"
    if durations:
        header += f"  {'hours':>8}"
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for index, (first, last) in enumerate(ranges):
        line = f"  {index:>5}  {first:>11}  {last:>11}  {last - first + 1:>9,}"
        if durations:
            line += f"  {sum(durations[first : last + 1]) / 3600:>7,.0f}h"
        print(line)
    covered = sum(last - first + 1 for first, last in ranges)
    print(f"\n  covers {covered:,} of {total:,} rows, no overlap, no gap")
    print("  machine-readable: add --shard-format tsv\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect flicker in egocentric videos")
    parser.add_argument(
        "video",
        nargs="?",
        help="One video: an s3://bucket/key URI, local path, or FFmpeg-supported URL",
    )
    parser.add_argument("--manifest", help="Manifest of videos to score (CSV or .txt list)")
    parser.add_argument(
        "--links",
        nargs="?",
        const="",
        help=(
            "Score every video named by a link sheet's video_s3_link column "
            "(.xlsx or .csv), against the configured s3.bucket. Pass no value to "
            "use s3.links from the config."
        ),
    )
    parser.add_argument(
        "--no-resolve-case",
        action="store_true",
        help=(
            "Treat the sheet's links as exact S3 keys and skip the bucket "
            "listing. The current sheet is lowercased and S3 keys are "
            "case-sensitive, so this makes every row 404; use it only once the "
            "sheet carries real keys."
        ),
    )
    parser.add_argument(
        "--link-report",
        default="reports/link_resolution.md",
        help="Where to list links that could not be resolved to a real object",
    )
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
    parser.add_argument(
        "--config",
        # Overridable by environment so an image or a host can set the
        # production config once, instead of every invocation repeating it
        # and one of them forgetting.
        default=os.environ.get("FLICKER_CONFIG", "configs/detector.yaml"),
        help="Detector config (default: $FLICKER_CONFIG or configs/detector.yaml)",
    )
    parser.add_argument("--logging-config", default="configs/logging.yaml")
    parser.add_argument(
        "--output",
        default="output/window_metrics.jsonl",
        help=(
            "Durable per-window output, JSON Lines: one line per video carrying "
            "every window's score and raw measurements. This is what --resume "
            "reads, and what the CSVs below are derived from."
        ),
    )
    parser.add_argument(
        "--video-csv",
        default="output/flag_manifest.csv",
        help=(
            "One row per video: the rollup in the original flag-manifest schema, "
            "which is what --calibrate-labels and the reporting scripts read."
        ),
    )
    parser.add_argument(
        "--window-dir",
        default="output/window_metrics",
        help=(
            "Root for per-video window metrics. Each video gets its own CSV at "
            "its key path, so raw/A/B/CLIP.MP4 lands at "
            "<root>/raw/A/B/CLIP.csv."
        ),
    )
    parser.add_argument(
        "--no-window-dir",
        action="store_true",
        help=(
            "Skip the per-video window CSVs. They can be rebuilt from the JSONL "
            "at any time with scripts/export_windows.py --window-dir."
        ),
    )
    parser.add_argument(
        "--no-video-csv",
        action="store_true",
        help="Skip the per-video rollup table",
    )
    parser.add_argument(
        "--s3-output",
        help=(
            "Also copy results to this s3://bucket/prefix, mirroring the local "
            "output layout. Defaults to s3.output in the config. Results are "
            "written locally first, so a failed upload never costs the run's "
            "work. Needs s3:PutObject on the prefix; nothing outside it is ever "
            "written, and nothing is deleted."
        ),
    )
    parser.add_argument(
        "--no-s3-output",
        action="store_true",
        help="Keep results local even though s3.output is set in the config",
    )
    parser.add_argument(
        "--s3-output-dry-run",
        action="store_true",
        help="Resolve and log every destination key without uploading anything",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Local directory the S3 layout mirrors (default: output)",
    )
    parser.add_argument(
        "--print-shards",
        type=int,
        metavar="N",
        help=(
            "Print the --from-row/--to-row pairs that split the input into N "
            "equal, non-overlapping shards, then exit without scoring. Hand-typed "
            "ranges are where fleet gaps and double-scoring come from."
        ),
    )
    parser.add_argument(
        "--shard-format",
        choices=("human", "tsv"),
        default="human",
        help=(
            "human (default) prints a table; tsv prints 'index<TAB>from<TAB>to' "
            "and nothing else, for a launcher to read in a loop"
        ),
    )
    parser.add_argument(
        "--shard-by",
        choices=("duration", "rows"),
        default="duration",
        help=(
            "How --print-shards balances the shards. duration (default) equalises "
            "hours of video, which is what equalises runtime; rows equalises the "
            "row count, which leaves the heaviest shard ~1.2x the lightest."
        ),
    )
    parser.add_argument(
        "--from-row",
        type=int,
        help=(
            "First input row to process, counted from zero and INCLUSIVE. Rows "
            "are the link sheet's (or manifest's) own rows, so --from-row 0 "
            "--to-row 99 and --from-row 100 --to-row 199 tile without overlap "
            "and can run on separate instances."
        ),
    )
    parser.add_argument(
        "--to-row",
        type=int,
        help="Last input row to process, INCLUSIVE. Omit to run to the end.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Process only the first N videos of the work set, applied after "
            "--from-row/--to-row (stable across --resume)"
        ),
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
    links_requested = arguments.links is not None
    input_count = (
        int(bool(arguments.video))
        + int(bool(arguments.manifest))
        + int(listing_requested)
        + int(links_requested)
    )
    inputs = "VIDEO, --manifest, --links, or --s3-prefix"
    if arguments.write_manifest and not (listing_requested or links_requested):
        parser.error("--write-manifest requires --s3-prefix or --links")
    if arguments.print_shards is not None and arguments.print_shards < 1:
        parser.error("--print-shards must be at least one")
    if arguments.calibrate_labels:
        if input_count:
            parser.error(f"calibration does not accept {inputs}")
    elif input_count != 1:
        parser.error(f"provide exactly one of {inputs}")
    # --output used to name the per-video CSV and now names the per-window JSONL.
    # Failing loudly beats writing JSON Lines into a file called .csv, which the
    # reporting scripts would then read as a malformed manifest.
    # --window-dir used to be --window-csv, one combined file.  A path that still
    # looks like a file is almost certainly the old flag, and creating a
    # directory called `window_metrics.csv` would be a confusing way to find out.
    try:
        validate_row_range(arguments.from_row, arguments.to_row)
    except ValueError as error:
        parser.error(str(error))
    if arguments.window_dir.lower().endswith(".csv"):
        parser.error(
            "--window-dir names a directory: window metrics are now one CSV per "
            "video at its key path. Pass a directory, e.g. output/window_metrics"
        )
    if not arguments.calibrate_labels and arguments.output.lower().endswith(".csv"):
        parser.error(
            "--output now names the per-window JSON Lines file (schema 2.0); "
            "use --video-csv for the per-video CSV rollup, e.g. "
            "--output output/window_metrics.jsonl --video-csv " + arguments.output
        )

    configure_logging(arguments.logging_config)
    config = _apply_decode_overrides(
        _apply_aws_overrides(load_config(arguments.config), arguments), arguments
    )
    video_csv = None if arguments.no_video_csv else arguments.video_csv
    window_dir = None if arguments.no_window_dir else arguments.window_dir
    publisher = None
    # Only the modes that actually score videos publish anything, and the write
    # check costs a request: --print-shards and --write-manifest upload nothing, so
    # requiring write credentials for them would block the very command an operator
    # runs first, before any credentials are arranged.
    scoring = not (arguments.print_shards or arguments.write_manifest or arguments.calibrate_labels)
    destination = (
        None if arguments.no_s3_output or not scoring else _s3_output(arguments, config)
    )
    if destination:
        settings = S3Settings.from_config(config)
        section = config.get("s3") or {}
        publisher = S3Publisher.from_uri(
            destination,
            # The output bucket may live in another region than the corpus, and
            # SigV4 is region-scoped, so it gets its own setting.
            region=section.get("output_region") or settings.region,
            profile=settings.profile,
            dry_run=arguments.s3_output_dry_run,
        )
        logger.info(
            "Publishing results to %s%s",
            publisher.uri,
            " (dry run)" if arguments.s3_output_dry_run else "",
        )
        # Checked before any decoding, so a read-only role stops the run at the
        # first request instead of after hours of work with nothing uploaded.
        publisher.verify_writable()
    try:
        if arguments.print_shards:
            print_shards(arguments, config)
            return
        if arguments.calibrate_labels:
            # Calibration is fitted on the video rollup, because the reference
            # labels are per video. Window-level labels would let the window
            # thresholds be fitted the same way; see decision.window.
            if video_csv is None:
                parser.error("calibration needs the per-video rollup; drop --no-video-csv")
            # Rank every signal before fitting anything. A threshold fit on an
            # inverted score still reports a number, so the ordering check has to
            # come first and be visible next to the fitted bands.
            discrimination = measure_discrimination(video_csv, arguments.calibrate_labels)
            write_discrimination_report(discrimination, arguments.discrimination_report)
            for measured in discrimination:
                print(
                    f"  {measured.signal:>22} spearman={measured.spearman:+.3f} "
                    f"AUC={measured.auc_extreme_vs_none:.3f}"
                    f"{'   INVERTED' if measured.is_inverted else ''}"
                )
            result = calibrate(video_csv, arguments.calibrate_labels)
            write_calibration_report(result, arguments.calibration_report)
            print(
                "Calibration complete: "
                f"mild={result.mild_threshold:.4f} extreme={result.extreme_threshold:.4f}"
            )
        elif arguments.write_manifest and links_requested:
            # Pin the resolved sheet: the real, correctly-cased keys, so later
            # runs neither re-list the bucket nor re-derive the resolution.
            written = write_manifest(
                link_entries(
                    _links_path(arguments, config, parser),
                    config,
                    resolve_case=not arguments.no_resolve_case,
                    report_path=arguments.link_report,
                ),
                arguments.write_manifest,
            )
            print(f"Wrote {written} resolved entries to {arguments.write_manifest}")
        elif arguments.write_manifest:
            written = snapshot_s3_manifest(
                arguments.s3_prefix,
                arguments.write_manifest,
                config,
                limit=arguments.limit,
            )
            print(f"Wrote {written} entries to {arguments.write_manifest}")
        elif links_requested:
            run_links(
                _links_path(arguments, config, parser),
                arguments.output,
                config,
                limit=arguments.limit,
                resume=arguments.resume,
                workers=arguments.workers,
                video_csv=video_csv,
                window_dir=window_dir,
                publisher=publisher,
                publish_root=arguments.output_root,
                resolve_case=not arguments.no_resolve_case,
                report_path=arguments.link_report,
                from_row=arguments.from_row,
                to_row=arguments.to_row,
            )
        elif listing_requested:
            run_s3_prefix(
                arguments.s3_prefix,
                arguments.output,
                config,
                limit=arguments.limit,
                resume=arguments.resume,
                workers=arguments.workers,
                video_csv=video_csv,
                window_dir=window_dir,
                publisher=publisher,
                publish_root=arguments.output_root,
            )
        elif arguments.manifest:
            run_manifest(
                arguments.manifest,
                arguments.output,
                config,
                limit=arguments.limit,
                resume=arguments.resume,
                workers=arguments.workers,
                video_csv=video_csv,
                window_dir=window_dir,
                publisher=publisher,
                publish_root=arguments.output_root,
                from_row=arguments.from_row,
                to_row=arguments.to_row,
            )
        else:
            video_result = process_one(arguments.video, config)
            write_result(
                video_result,
                arguments.output,
                video_csv=video_csv,
                window_dir=window_dir,
                publisher=publisher,
                publish_root=arguments.output_root,
            )
            _print_window_summary(video_result)
    except S3AccessError as error:
        logger.error("S3 access failed: %s", error)
        raise SystemExit(2) from error
    except Exception:
        logger.exception("Flicker detection failed")
        raise


def _s3_output(arguments: argparse.Namespace, config: dict[str, Any]) -> str | None:
    """Resolve the upload destination: the flag if given, else ``s3.output``."""
    if arguments.s3_output:
        return str(arguments.s3_output)
    configured = (config.get("s3") or {}).get("output")
    return str(configured) if configured else None


def _links_path(
    arguments: argparse.Namespace,
    config: dict[str, Any],
    parser: argparse.ArgumentParser | None = None,
) -> str:
    """Resolve ``--links`` to a path, falling back to ``s3.links`` in the config."""
    if arguments.links:
        return str(arguments.links)
    configured = (config.get("s3") or {}).get("links")
    if not configured:
        message = "--links needs a path, or s3.links set in the detector config"
        if parser is not None:
            parser.error(message)
        raise SystemExit(message)
    return str(configured)


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
