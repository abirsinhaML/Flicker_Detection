"""Score the same videos on both decode backends and report the difference.

Hardware and software decode are not bit-identical: NVDEC applies its own
scaler, not swscale's.  The scores are therefore expected to differ slightly,
and the question this script answers is whether they differ by enough to matter
to a calibrated gate whose mild/extreme boundaries sit at 0.328 and 0.51.

Run it before switching a batch to ``decode.backend: cuda``, and again on real
corpus videos rather than only on synthetic ones -- a fisheye scene resamples
differently from a test pattern.

    uv run python scripts/compare_decode_backends.py path/to/clip.mp4
    uv run python scripts/compare_decode_backends.py s3://bucket/key.mp4 \
        --config configs/detector_1.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import load_config, process_one, process_video  # noqa: E402
from src.core.types import VideoResult  # noqa: E402
from src.data.s3_source import redact_url  # noqa: E402
from src.utils.logger import configure_logging  # noqa: E402

# The routing decision is what a difference has to survive; the detector scores
# are reported so a drift can be attributed to the channel that caused it.
_COMPARED = (
    "flicker_score",
    "illuminant_score",
    "rolling_band_score",
    "awb_score",
    "horizontal_coherence",
    "valid_fraction",
)


def _score(
    source: str,
    config: dict,
    backend: str,
    *,
    gpu_resize: bool = False,
) -> tuple[VideoResult, float]:
    scoped = copy.deepcopy(config)
    scoped["decode"] = {**(scoped.get("decode") or {}), "backend": backend}
    # Insisting on the backend is the point: a silent fallback would compare
    # software against software and report a reassuring zero.
    if backend != "cpu":
        scoped["decode"]["allow_fallback"] = False
        scoped["decode"]["gpu_resize"] = gpu_resize

    started = perf_counter()
    if source.startswith("s3://") or "://" in source:
        result = process_one(source, scoped)
    else:
        result = process_video(source, scoped)
    return result, perf_counter() - started


def _detector_value(result: VideoResult, field: str) -> float:
    if field.endswith("_score") and field != "flicker_score":
        mapping = {
            "illuminant_score": "IlluminantDetector",
            "rolling_band_score": "RollingBandDetector",
            "awb_score": "AWBDetector",
        }
        return float(result.detector_scores.get(mapping[field], 0.0))
    return float(getattr(result, field))


def compare(source: str, config: dict, *, gpu_resize: bool = False) -> bool:
    """Score one video both ways; return whether the routing decision held."""
    print(f"\n{'=' * 74}\n  {redact_url(source)}  (gpu_resize={gpu_resize})\n{'=' * 74}")

    cpu_result, cpu_seconds = _score(source, config, "cpu")
    gpu_result, gpu_seconds = _score(source, config, "cuda", gpu_resize=gpu_resize)

    print(f"  {'field':<24}{'cpu':>14}{'cuda':>14}{'delta':>14}")
    worst = 0.0
    for field in _COMPARED:
        left, right = _detector_value(cpu_result, field), _detector_value(gpu_result, field)
        worst = max(worst, abs(left - right))
        print(f"  {field:<24}{left:>14.6f}{right:>14.6f}{right - left:>+14.6f}")

    agreed = (
        cpu_result.route == gpu_result.route
        and cpu_result.severity_band == gpu_result.severity_band
    )
    verdict = "ok" if agreed else "DIVERGED"
    print(f"  {'route':<24}{cpu_result.route:>14}{gpu_result.route:>14}{verdict:>14}")
    elapsed_delta = gpu_seconds - cpu_seconds
    print(f"  {'wall seconds':<24}{cpu_seconds:>14.1f}{gpu_seconds:>14.1f}{elapsed_delta:>+14.1f}")
    print(f"  {'speedup':<24}{'':>14}{'':>14}{cpu_seconds / max(gpu_seconds, 1e-9):>13.2f}x")
    print(f"  worst absolute score delta: {worst:.6f}")
    return agreed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", help="Local paths or s3:// URIs")
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument("--logging-config", default="configs/logging.yaml")
    parser.add_argument(
        "--gpu-resize",
        action="store_true",
        help="Let NVDEC apply the downscale too, instead of leaving it to swscale",
    )
    arguments = parser.parse_args()

    configure_logging(arguments.logging_config)
    config = load_config(arguments.config)

    diverged = [
        source
        for source in arguments.sources
        if not compare(source, config, gpu_resize=arguments.gpu_resize)
    ]
    print()
    if diverged:
        print(f"{len(diverged)} of {len(arguments.sources)} videos changed routing decision:")
        for source in diverged:
            print(f"  {redact_url(source)}")
        raise SystemExit(1)
    print(f"All {len(arguments.sources)} videos kept the same severity band and route.")


if __name__ == "__main__":
    main()
