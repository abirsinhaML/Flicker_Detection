"""Explain a low ``valid_fraction``: dark footage, or a decode that returned black.

``AnalysisRegion.detect`` warns when a window's live pixels fall under
``min_valid_fraction`` and then measures the whole frame instead.  The warning
does not say *why* the window looked dead, and the ``valid_fraction`` column
cannot answer it either -- the fallback overwrites the measured fraction with
the full-frame one, so a blacked-out window and a healthy one both report 0.75.

Two causes need telling apart:

  * the footage really is that dark (camera bagged, unlit room, night shift), in
    which case the fallback is doing its job; or
  * the decode returned black or torn frames for that window, in which case the
    score built on it is meaningless and the fallback is hiding a decode bug.

Decoding each window on both backends separates them.  If software decode shows
the same darkness, it is the footage.  If software decode shows a normal frame
where NVDEC returned black, it is the hardware path.

    uv run python scripts/diagnose_dark_windows.py s3://bucket/key.MP4
    uv run python scripts/diagnose_dark_windows.py raw/prefix/GX010011.MP4 \
        --config configs/detector_1.yaml --max-windows 20
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _parse_resize, load_config  # noqa: E402
from src.data.decode import DecodePolicy  # noqa: E402
from src.data.reader import VideoReader  # noqa: E402
from src.data.s3_source import S3Settings, redact_url  # noqa: E402
from src.data.sampler import WindowSampler  # noqa: E402
from src.signals.mask import RegionPolicy  # noqa: E402
from src.signals.preprocessing import SignalPreprocessor  # noqa: E402


@dataclass(frozen=True, slots=True)
class WindowStats:
    """What one window looked like to the region detector."""

    frames: int
    live_fraction: float
    black_frames: int
    luma_max: float
    luma_mean: float
    luma_p99: float

    @property
    def is_all_black(self) -> bool:
        return self.frames > 0 and self.black_frames == self.frames


def _stats(frames: np.ndarray, policy: RegionPolicy) -> WindowStats:
    """Measure a window the way ``AnalysisRegion.detect`` sees it."""
    if not len(frames):
        return WindowStats(0, 0.0, 0, 0.0, 0.0, 0.0)

    luma = SignalPreprocessor.rgb_to_yuv(frames)[..., 0]
    # The same test the region detector applies: a pixel is live if it ever
    # exceeds the border threshold, judged over the whole window.
    live = luma.max(axis=0) > policy.border_threshold
    per_frame_peak = luma.reshape(len(luma), -1).max(axis=1)

    return WindowStats(
        frames=len(frames),
        live_fraction=float(live.mean()),
        black_frames=int((per_frame_peak <= policy.border_threshold).sum()),
        luma_max=float(luma.max()),
        luma_mean=float(luma.mean()),
        luma_p99=float(np.percentile(luma, 99)),
    )


def _scan(source: str, config: dict, backend: str, limit: int | None) -> dict[float, WindowStats]:
    """Walk the configured window schedule on one decode backend."""
    scoped = copy.deepcopy(config)
    scoped["decode"] = {**(scoped.get("decode") or {}), "backend": backend}
    # Insisting on the backend is the point: a silent fallback to software would
    # compare software against software and clear NVDEC of a fault it caused.
    if backend != "cpu":
        scoped["decode"]["allow_fallback"] = False

    sampling = scoped["sampling"]
    sampler = WindowSampler(sampling["window_duration"], sampling["stride"])
    policy = RegionPolicy.from_config(scoped.get("analysis_region"))
    resolver = S3Settings.from_config(scoped).resolver()

    measured: dict[float, WindowStats] = {}
    with VideoReader(
        resolver.resolve(source),
        decode_policy=DecodePolicy.from_config(scoped.get("decode")),
        resize=_parse_resize(sampling.get("resize")),
    ) as reader:
        metadata = reader.metadata()
        print(
            f"  backend={reader.decode_backend} duration={metadata.duration:.2f}s "
            f"{metadata.width}x{metadata.height} @ {metadata.fps:.3f}fps"
        )
        for index, spec in enumerate(sampler.sample(metadata.duration)):
            if limit is not None and index >= limit:
                break
            window = reader.read_window(
                start=spec.start_time,
                duration=spec.end_time - spec.start_time,
            )
            measured[round(spec.start_time, 3)] = _stats(window.frames, policy)
    return measured


def _verdict(cuda: WindowStats, cpu: WindowStats, policy: RegionPolicy) -> str:
    """Attribute one window's darkness to the footage or to the decoder."""
    cuda_dark = cuda.live_fraction < policy.min_valid_fraction
    cpu_dark = cpu.live_fraction < policy.min_valid_fraction

    if not cuda.frames and not cpu.frames:
        return "NO FRAMES on either backend"
    if not cuda.frames:
        return "DECODE: cuda returned no frames"
    if cuda_dark and not cpu_dark:
        return "DECODE: cuda black, cpu fine"
    if cuda_dark and cpu_dark:
        return "footage: dark on both backends"
    if cpu_dark and not cuda_dark:
        return "DECODE: cpu black, cuda fine"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="s3:// URI, bare key, or local path")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--logging-config", default=None, help="omit to keep the report clean")
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--backend",
        default="cuda",
        help="the backend under suspicion; always compared against cpu",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    policy = RegionPolicy.from_config(config.get("analysis_region"))

    source = arguments.source
    if not (source.startswith("s3://") or "://" in source or Path(source).exists()):
        bucket = (config.get("s3") or {}).get("bucket")
        if not bucket:
            parser.error("bare keys need an s3.bucket in the config")
        source = f"s3://{bucket}/{source.lstrip('/')}"

    print(f"source: {redact_url(source)}")
    print(f"policy: border_threshold={policy.border_threshold} "
          f"min_valid_fraction={policy.min_valid_fraction} "
          f"exclude_bottom_fraction={policy.exclude_bottom_fraction}")
    print()

    print(f"scanning on {arguments.backend}:")
    hardware = _scan(source, config, arguments.backend, arguments.max_windows)
    print(f"scanning on cpu:")
    software = _scan(source, config, "cpu", arguments.max_windows)
    print()

    header = (
        f"{'start':>9}  {'frames':>13}  {'live %':>13}  "
        f"{'luma max':>13}  {'luma mean':>13}  verdict"
    )
    print(header)
    print("-" * len(header))

    causes: dict[str, int] = {}
    for start in sorted(set(hardware) | set(software)):
        empty = WindowStats(0, 0.0, 0, 0.0, 0.0, 0.0)
        gpu, soft = hardware.get(start, empty), software.get(start, empty)
        verdict = _verdict(gpu, soft, policy)
        causes[verdict] = causes.get(verdict, 0) + 1
        flag = " " if verdict == "ok" else "*"
        print(
            f"{start:>9.1f}  {gpu.frames:>6d}/{soft.frames:<6d}  "
            f"{gpu.live_fraction * 100:>6.1f}/{soft.live_fraction * 100:<6.1f}  "
            f"{gpu.luma_max:>6.0f}/{soft.luma_max:<6.0f}  "
            f"{gpu.luma_mean:>6.1f}/{soft.luma_mean:<6.1f}  {flag}{verdict}"
        )

    print()
    print("summary (gpu/cpu shown side by side):")
    for cause, count in sorted(causes.items(), key=lambda item: -item[1]):
        print(f"  {count:>4d}  {cause}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
