#!/usr/bin/env python3
"""Fit the none/mild boundary against BurstFlicker-G ground truth.

BurstFlicker-G ships matched pairs: ``gt/NNNN.mp4`` is flicker-free and
``flicker/NNNN.mp4`` is the same scene with flicker.  That binary split maps
exactly onto the one boundary this script fits -- ``none`` versus not-``none`` --
and unlike the VisionLabs colour-coded sheet it is construction, not judgement,
so there is no label noise to argue about.

It deliberately says nothing about ``extreme_threshold``: BurstFlicker has no
severity grades, so the mild/extreme split cannot be fitted from it and is left
where the config puts it.

What the output is *not*: a production threshold.  Two effects separate a
0.33 s BurstFlicker burst from a 3 s production window, both measurable without
labels, and the report prints the scaled equivalent alongside the raw fit so the
gap is visible rather than assumed.  See the ``WINDOW_LENGTH_FACTOR`` note.

Usage:
    python scripts/fit_none_threshold.py
    python scripts/fit_none_threshold.py --limit 40 --workers 4
    python scripts/fit_none_threshold.py --weights 0.10,0.65,0.25
    python scripts/fit_none_threshold.py --scores output/burstflicker/pair_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_gt_frames import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    REGION_MASKED,
    REGION_POLICIES,
    GroundTruthClip,
    _apply_weight_override,
    load_config,
    safe_worker_count,
    score_clip,
)

logger = logging.getLogger(__name__)

# Measured, not assumed: identical synthetic content with an identical imposed
# sinusoidal modulation at 9.0 Hz (an exact PSD bin at both lengths) scores this
# much higher over 90 frames than over 10.  `rolling_band_score` is invariant to
# the change to five decimals; the factor is carried entirely by the illuminant
# and AWB channels, whose prominence and peak-to-median measurements need bins to
# resolve against.
#
# It therefore depends on the weights, and must be re-measured when they move:
# 1.40x at 0.30/0.50/0.20, 1.25x at 0.10/0.65/0.25, because shifting weight onto
# the invariant channel shrinks it.  Weighting rolling-band alone would make the
# threshold window-length independent and remove this correction entirely.
WINDOW_LENGTH_FACTOR = 1.25

SCORE_FIELDS = (
    "clip_key",
    "split",
    "scene",
    "kind",
    "has_flicker",
    "flicker_score",
    "illuminant_score",
    "rolling_band_score",
    "awb_score",
)


@dataclass(frozen=True, slots=True)
class Operating:
    """One candidate threshold and what it does to both ground-truth classes."""

    threshold: float
    false_positive_rate: float
    recall: float

    @property
    def youden_j(self) -> float:
        return self.recall - self.false_positive_rate


def discover_pairs(dataset_root: Path, splits: list[str]) -> list[GroundTruthClip]:
    """Find every ``gt/`` and ``flicker/`` clip, tagging the kind in the key."""
    clips: list[GroundTruthClip] = []
    for split in splits:
        for kind in ("gt", "flicker"):
            directory = dataset_root / split / kind
            if not directory.is_dir():
                raise FileNotFoundError(f"No directory at {directory}")
            for path in sorted(directory.glob("*.mp4")):
                clips.append(
                    GroundTruthClip(
                        key=f"{split}/{kind}/{path.stem}",
                        split=split,
                        scene=path.stem,
                        path=path,
                    )
                )
    if not clips:
        raise FileNotFoundError(f"No clips under {dataset_root} for splits {splits}")
    return clips


def score_pairs(
    clips: list[GroundTruthClip],
    config: dict[str, Any],
    *,
    workers: int,
) -> list[dict[str, object]]:
    """Score every clip and keep one clip-level row each.

    ``score_clip`` is reused rather than reimplemented so both halves of the pair
    go through exactly the same path the delivered per-frame audit uses; the
    per-frame rows it returns are collapsed to their first, which carries the
    clip-level verdict.
    """
    rows: list[dict[str, object]] = []
    failures = 0

    def collapse(clip: GroundTruthClip, frame_rows: list[dict[str, object]]) -> None:
        nonlocal failures
        head = frame_rows[0] if frame_rows else {}
        if head.get("status") != "ok":
            failures += 1
            logger.warning("Skipping %s: %s", clip.key, head.get("error", "no rows"))
            return
        kind = "flicker" if "/flicker/" in clip.key else "gt"
        rows.append(
            {
                "clip_key": clip.key,
                "split": clip.split,
                "scene": clip.scene,
                "kind": kind,
                "has_flicker": int(kind == "flicker"),
                "flicker_score": head["flicker_score"],
                "illuminant_score": head["illuminant_score"],
                "rolling_band_score": head["rolling_band_score"],
                "awb_score": head["awb_score"],
            }
        )

    if workers <= 1:
        for clip in clips:
            collapse(clip, score_clip(clip, config))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(score_clip, clip, config): clip for clip in clips}
            for done, future in enumerate(as_completed(futures), start=1):
                clip = futures[future]
                try:
                    collapse(clip, future.result())
                except Exception:
                    failures += 1
                    logger.exception("Failed clip: %s", clip.key)
                if done % 100 == 0:
                    logger.info("Progress: %d/%d clips", done, len(clips))

    if failures:
        logger.warning("%d clips produced no usable score and are excluded", failures)
    rows.sort(key=lambda row: str(row["clip_key"]))
    return rows


def operating_curve(clean: np.ndarray, flicker: np.ndarray) -> list[Operating]:
    """Evaluate every threshold that can change a decision.

    Candidates are the observed scores themselves rather than a fixed grid, so no
    achievable operating point is missed between grid steps and none is counted
    twice.
    """
    candidates = np.unique(np.concatenate([clean, flicker, [0.0]]))
    return [
        Operating(
            threshold=float(t),
            false_positive_rate=float(np.mean(clean >= t)),
            recall=float(np.mean(flicker >= t)),
        )
        for t in candidates
    ]


def roc_auc(clean: np.ndarray, flicker: np.ndarray) -> float:
    """Rank-based AUC with ties counted as half, so exact zeros do not inflate it."""
    if not clean.size or not flicker.size:
        return float("nan")
    order = np.argsort(np.concatenate([clean, flicker]), kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    values = np.concatenate([clean, flicker])[order]
    index = 0
    while index < values.size:
        stop = index
        while stop + 1 < values.size and values[stop + 1] == values[index]:
            stop += 1
        ranks[order[index : stop + 1]] = 0.5 * (index + stop) + 1.0
        index = stop + 1
    positive_ranks = ranks[clean.size :].sum()
    n_pos, n_neg = flicker.size, clean.size
    return float((positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def report(rows: list[dict[str, object]], config: dict[str, Any], probe: float) -> None:
    """Print the operating curve and the thresholds worth considering."""
    clean = np.array([float(str(r["flicker_score"])) for r in rows if not r["has_flicker"]])
    flicker = np.array([float(str(r["flicker_score"])) for r in rows if r["has_flicker"]])
    weights = config["aggregation"]["weights"]
    committed = float(config["decision"]["mild_threshold"])

    print(f"\n{'=' * 78}")
    print("  NONE / MILD BOUNDARY, FITTED ON BURSTFLICKER-G GROUND TRUTH")
    print(f"{'=' * 78}")
    print(
        f"  weights: illuminant={weights['IlluminantDetector']:.2f} "
        f"rolling={weights['RollingBandDetector']:.2f} awb={weights['AWBDetector']:.2f}"
        f"      committed mild_threshold={committed:.4f}"
    )
    print(
        f"  clean (gt) n={clean.size}     flicker n={flicker.size}"
        f"     ROC AUC={roc_auc(clean, flicker):.4f}"
    )
    for name, values in (("clean  ", clean), ("flicker", flicker)):
        print(
            f"  {name} min={values.min():.4f} p05={np.percentile(values, 5):.4f} "
            f"median={np.median(values):.4f} p95={np.percentile(values, 95):.4f} "
            f"p99={np.percentile(values, 99):.4f} max={values.max():.4f}"
        )

    # The recall ceiling is the headline constraint and is independent of any
    # threshold: a flickering clip that scores zero cannot be separated from a
    # clean one that also scores zero, at any boundary.
    reachable = int(np.sum(flicker > clean.min()))
    dead = int(np.sum(flicker <= clean.min()))
    print(
        f"\n  RECALL CEILING: {reachable}/{flicker.size} "
        f"({reachable / flicker.size:.1%}) flickering clips score above the "
        f"cleanest clean clip.\n  {dead} score at or below it, so no threshold "
        f"can reach them."
    )

    # Evaluated at the requested thresholds rather than snapped to the nearest
    # observed score: both rates are well defined at any threshold, and snapping
    # made several requested rows collapse onto one candidate.
    curve = operating_curve(clean, flicker)
    targets = sorted(
        {0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, committed, probe}
    )
    print(f"\n{'threshold':>10} {'FP rate (clean)':>16} {'recall (flicker)':>17} {'Youden J':>10}")
    print("-" * 58)
    for target in targets:
        point = Operating(
            threshold=target,
            false_positive_rate=float(np.mean(clean >= target)),
            recall=float(np.mean(flicker >= target)),
        )
        marks = []
        if abs(target - committed) < 1e-9:
            marks.append("committed")
        if abs(target - probe) < 1e-9:
            marks.append("probe")
        suffix = f"  <- {', '.join(marks)}" if marks else ""
        print(
            f"{point.threshold:>10.4f} {point.false_positive_rate:>15.1%} "
            f"{point.recall:>16.1%} {point.youden_j:>10.3f}{suffix}"
        )

    print("\n  Candidate operating points:")
    zero_fp = max((p for p in curve if p.false_positive_rate == 0.0), key=lambda p: p.recall)
    print(
        f"    zero false positives   threshold >= {zero_fp.threshold:.4f}  "
        f"recall {zero_fp.recall:.1%}   (highest clean clip is {clean.max():.4f})"
    )
    for budget in (0.01, 0.05):
        best = max(
            (p for p in curve if p.false_positive_rate <= budget),
            key=lambda p: p.recall,
        )
        print(
            f"    FP rate <= {budget:.0%}           threshold >= {best.threshold:.4f}  "
            f"recall {best.recall:.1%}"
        )
    best_j = max(curve, key=lambda p: p.youden_j)
    print(
        f"    max Youden J           threshold  = {best_j.threshold:.4f}  "
        f"recall {best_j.recall:.1%}  FP {best_j.false_positive_rate:.1%}  J={best_j.youden_j:.3f}"
    )

    print(f"\n  Scaled to a 3 s production window (x{WINDOW_LENGTH_FACTOR}, measured):")
    for name, point in (("zero-FP", zero_fp), ("max Youden J", best_j)):
        scaled = point.threshold * WINDOW_LENGTH_FACTOR
        print(f"    {name:<14} {point.threshold:.4f} -> {scaled:.4f}")
    print(f"{'=' * 78}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the none/mild boundary on BurstFlicker-G ground truth",
    )
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--config", default="configs/detector.yaml")
    parser.add_argument(
        "--scores",
        default="output/burstflicker/pair_scores.csv",
        help="Where to write the per-clip scores of both classes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Score only the first N clips per split and kind (for a quick check)",
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument("--region", choices=REGION_POLICIES, default=REGION_MASKED)
    parser.add_argument("--weights", help="Override weights as illuminant,rolling,awb")
    parser.add_argument(
        "--probe",
        type=float,
        default=0.15,
        help="Highlight this candidate threshold in the curve (default: 0.15)",
    )
    parser.add_argument(
        "--from-scores",
        action="store_true",
        help=(
            "Re-print the report from an existing --scores file instead of "
            "rescoring. Decoding 738 clips at 6960x4640 takes minutes; exploring "
            "thresholds or a changed WINDOW_LENGTH_FACTOR should not."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = _apply_weight_override(
        load_config(Path(arguments.config), region=arguments.region),
        arguments.weights,
    )
    scores_path = Path(arguments.scores)
    if arguments.from_scores:
        with scores_path.open(newline="", encoding="utf-8") as handle:
            cached = [
                {**row, "has_flicker": int(row["has_flicker"])} for row in csv.DictReader(handle)
            ]
        if not cached:
            raise SystemExit(f"No rows in {scores_path}")
        logger.info("Reporting on %d cached clip scores from %s", len(cached), scores_path)
        report(cached, config, arguments.probe)
        return

    clips = discover_pairs(Path(arguments.dataset_root), arguments.splits)
    if arguments.limit is not None:
        kept: dict[tuple[str, str], int] = {}
        selected = []
        for clip in clips:
            kind = "flicker" if "/flicker/" in clip.key else "gt"
            bucket = (clip.split, kind)
            if kept.get(bucket, 0) < arguments.limit:
                kept[bucket] = kept.get(bucket, 0) + 1
                selected.append(clip)
        clips = selected

    workers = safe_worker_count(max(arguments.workers, 1), clips[0].path)
    logger.info("Scoring %d clips (both classes) with %d workers", len(clips), workers)
    rows = score_pairs(clips, config, workers=workers)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d clip scores to %s", len(rows), scores_path)

    report(rows, config, arguments.probe)


if __name__ == "__main__":
    main()
