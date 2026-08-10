# Fitting the none/mild boundary on ground truth

## The question

`decision.mild_threshold` in `configs/detector.yaml` decides which videos are
accepted unseen and which go to a human. It was originally set at 0.328 against
the 25 colour-coded VisionLabs reference videos, on the grounds that no reference
video scored between 0.3195 (highest clean) and 0.3365 (lowest mild), so the
midpoint separated those two classes perfectly.

Two objections to that reasoning drove this work:

1. A 0.017-wide empty interval across 17 videos is what you would expect even if
   the classes fully overlap. It is evidence that nothing landed there, not
   evidence that nothing can. The threshold is fitted to n=17 across the
   boundary and its confidence interval is wide.
2. **The VisionLabs labels are not ground truth.** They are a human colour-code
   from a delivery spreadsheet — one reviewer's eye. The repo already treats the
   sheet as fallible: `methodology.md` notes its orange-vs-red ambiguity, and the
   `extreme_threshold` comment records a deliberate disagreement with it on four
   videos.

BurstFlicker-G is different in kind: it ships matched pairs, so the clean/flicker
distinction is construction rather than judgement. This document fits the
none/mild boundary against it.

It says nothing about `extreme_threshold`. BurstFlicker has no severity grades,
so the mild/extreme split cannot be fitted from it.

## The datasets

### BurstFlicker (Kaggle `lishenqu/burstflicker`, version 1)

**BurstFlicker-G** — the set used here. 369 matched scenes (338 train, 31 test).
Each scene is a pair: `gt/NNNN.mp4` is flicker-free, `flicker/NNNN.mp4` is the
same scene with flicker. 10 frames at 30 fps (0.333 s), 6960×4640.

Two clips are shorter than the rest in the source: `train/gt/0040` has 7 frames
and `train/gt/0136` has 6. That is why the per-frame audit has 3,683 rows rather
than 369 × 10 = 3,690.

The flicker is *generated*, not captured. That is the main reason to treat what
follows as a strong prior rather than a final answer.

**BurstFlicker-S** — real captures as JPEG sequences: `input/` holds 10 flicker
frames per scene, `gt/` holds 2 flicker-free frames. 400 scenes, rebuilt locally
as `data/burstflicker/s_clips/*/{gt,flicker}/*.mp4` at 1080×720.

**BurstFlicker-S `gt/` is unusable for this fit.** Two frames cannot support a
Welch PSD — `FrequencyAnalyzer.select_peak` requires at least three spectral bins
and returns an empty peak below that — so every detector reads exactly 0.0 and
every clip scores 0.0 by construction. This is visible in
`output/burstflicker/s_scores.csv` and is not a property of the footage.

### VisionLabs reference corpus (for contrast only)

25 videos, one factory, two operators, ~5 minutes each, 1280×720 raw fisheye (a
960×720 image circle pillarboxed, so 25% of every frame is matte black). Labels:
7 `none`, 10 `mild`, 8 `extreme`, assigned by eye.

## What "clean" measures on ground truth

`scripts/score_gt_frames.py` scores all 369 `gt/` clips — the false-positive
audit, since every one of them is flicker-free by construction. Run under both
weight sets:

| weights (illum/roll/awb) | min | median | mean | p95 | p99 | **max** | false positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.30 / 0.50 / 0.20 | 0.0000 | 0.0005 | 0.0034 | 0.0148 | 0.0496 | **0.1078** | 0 / 369 |
| 0.10 / 0.65 / 0.25 | 0.0000 | 0.0006 | 0.0039 | 0.0182 | 0.0618 | **0.1291** | 0 / 369 |

117 clips score exactly 0.0 at the first weight set. Nothing reaches `mild` under
either.

The clean distribution rose 1.16× at the mean and 1.20× at the max when weight
moved off `illuminant` (mean 0.0015 here, the smallest of the three) onto
`rolling_band` (0.0048, the largest) and `awb` (0.0027); 219 of 369 clips rose.
Raising `mild_threshold` from 0.328 to 0.365 covered it, though headroom narrowed
slightly: 0.328/0.1078 = 3.0× before, 0.365/0.1291 = 2.8× after.

**This establishes that the detector has no intrinsic false-positive floor.** On
genuinely clean, largely motion-free footage it reads 0.000–0.129. The score is
not structurally inflated. That is a fact about the detector, and it is the
foundation for everything below — but on its own it says nothing about where the
threshold belongs, because it only measures one side of the boundary.

Two incidental observations from the audit: `valid_fraction` ranges 0.2102–0.7500
(median 0.7447, the 0.75 being the fisheye bottom-25% exclusion applied to
non-fisheye footage), and two very dark clips tripped `min_valid_fraction` and
fell back to measuring the whole frame, which is logged.

## The operating curve

`scripts/fit_none_threshold.py` scores both halves of every pair — 738 clips — and
sweeps the boundary. At weights 0.10 / 0.65 / 0.25:

**ROC AUC 0.9578.**

| class | min | p05 | median | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean (`gt`, n=369) | 0.0000 | 0.0000 | 0.0006 | 0.0182 | 0.0618 | 0.1291 |
| flicker (n=369) | 0.0000 | 0.0058 | 0.2399 | 0.5681 | 0.6146 | 0.6518 |

| threshold | FP rate (clean) | recall (flicker) | Youden J | |
| ---: | ---: | ---: | ---: | --- |
| 0.0000 | 100.0% | 100.0% | 0.000 | |
| 0.0100 | 8.1% | 93.2% | 0.851 | |
| 0.0200 | 4.6% | 90.2% | 0.856 | |
| **0.0209** | **3.8%** | **89.7%** | 0.859 | ≈ max Youden J |
| 0.0500 | 1.4% | 78.0% | 0.767 | |
| 0.1000 | 0.5% | 63.4% | 0.629 | |
| **0.1311** | **0.0%** | **61.5%** | 0.615 | zero-FP point |
| 0.1500 | 0.0% | 58.5% | 0.585 | candidate considered |
| 0.2000 | 0.0% | 52.6% | 0.526 | |
| 0.2500 | 0.0% | 49.3% | 0.493 | |
| 0.3000 | 0.0% | 42.8% | 0.428 | |
| **0.3650** | **0.0%** | **33.6%** | 0.336 | committed |
| 0.4000 | 0.0% | 29.0% | 0.290 | |
| 0.5000 | 0.0% | 13.6% | 0.136 | |

Candidate operating points:

| criterion | threshold | FP rate | recall |
| --- | ---: | ---: | ---: |
| zero false positives | ≥ 0.1311 | 0.00% | 61.5% |
| FP rate ≤ 1% | ≥ 0.0768 | 0.81% | 69.7% |
| FP rate ≤ 5% | ≥ 0.0183 | 4.88% | 90.8% |
| max Youden J | 0.020880 | 3.79% | 90.0% |

Thresholds in the curve table are evaluated at the stated value, not snapped to
the nearest observed score; both rates are step functions and well defined
anywhere. The optima are searched over the observed scores themselves, so no
achievable operating point is missed between grid steps.

The two tables are consistent but must be read carefully at the optimum. The
Youden maximum sits at the observed score 0.020880, where recall is 90.0%.
Rounding a configured threshold *up* to 0.0209 steps past one flicker clip and
costs 0.3 points of recall, which is why the curve row above reads 89.7%. Round
down (0.02) rather than up if you configure a short value.

### Reading it

Three things follow.

**The current 0.365 is far too high.** It catches 33.6% of ground-truth flicker.
Dropping to 0.1311 nearly doubles that to 61.5% while still flagging none of the
369 clean clips. That gain is free — there is no false-positive cost to pay for
it, which makes the drop unambiguous.

**0.15 overshoots the safe point.** The highest clean clip is 0.1291, so zero
false positives is already achieved at 0.1311. Setting 0.15 gives away 3 points
of recall (61.5% → 58.5%) in exchange for nothing.

**Zero false positives is an expensive place to sit.** The FP curve is very flat
near the origin. Moving from 0.1311 down to 0.0209 costs 3.8% false positives and
buys 28.5 points of recall. For a QC gate the two errors are not symmetric: a
miss ships bad footage unseen, while a false positive costs review time on
footage a human still sees. Defending the last 4% of specificity is the wrong
side of that trade.

## Is the flicker we would miss actually mild?

Recall alone would be misleading if the missed clips were all barely-there
flicker — a gate arguably *should* pass the faintest cases. `output/burstflicker/g_severity.csv`
carries an independent physical severity measure per scene (`band_depth`,
computed from the pair difference, not from the model), so this is checkable.

| threshold | missed | median `band_depth` | caught | median `band_depth` | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1311 | 142 | 0.0330 | 227 | 0.0939 | 2.8× |
| 0.1500 | 153 | 0.0330 | 216 | 0.0966 | 2.9× |
| 0.3650 | 245 | 0.0398 | 124 | 0.1420 | 3.6× |

So the misses *are* systematically fainter — about 2.9× shallower at 0.15. That
genuinely softens the case for dropping further, and is the strongest argument
available for a conservative threshold.

It does not settle it, because the sorting is loose:
`spearman(flicker_score, band_depth) = +0.657` (and only +0.278 against
`temporal_depth`). A correlation of 0.657 means plenty of real flicker is missed
for reasons other than being mild.

## The 14 unreachable clips

**14 flicker clips score exactly 0.0000.** Since the cleanest clean clip also
scores 0.0000, no threshold anywhere can separate them, which caps recall at
**96.2%** (355/369) regardless of where the boundary goes.

These are not marginal cases:

| | median `band_depth` | range |
| --- | ---: | --- |
| the 14 zero-scoring clips | **0.1742** | 0.0185 – 0.4401 |
| all 369 flicker clips | 0.0585 | 0.0000 – 0.5215 |

Their median banding depth is **2.98× the flicker-class median**, reaching 0.44 —
among the strongest banding in the dataset, scoring zero. That is a detector
failure, not a threshold problem, and it is worth investigating independently of
this decision. No threshold choice can compensate for it.

## Transferring the fit to production

A threshold fitted on 0.33 s BurstFlicker bursts does not apply unchanged to
3-second windows of ~5-minute fisheye video. Three effects separate the domains.

### 1. Window length — measured, correctable

A 0.33 s burst gives 10 samples: 6 PSD bins, 5 above the 2 Hz `flicker_band`
floor, 3.00 Hz resolution. A 3 s production window gives 90 samples: 46 bins, 40
above the floor, 0.33 Hz. `peak_prominence` and `peak_to_median_ratio` are
measured against the spectrum's own neighbourhood and median, and with 5 bins the
median is largely the peak itself, so both are structurally compressed.

Measured directly with no labels involved: a real BurstFlicker `gt` frame at
320×180 as spatial content, with an imposed modulation
`base × (1 + depth × cos(2π(f·t + k·r)))` at f = 9.0 Hz — an exact PSD bin at
*both* 10 and 90 frames, so no result is an artifact of a peak falling between
bins — and k = 0.5 cycles per frame height.

At weights 0.10 / 0.65 / 0.25:

| depth | 0.33 s (10 fr) | 3.00 s (90 fr) | ratio | `rolling_band` 10 fr | `rolling_band` 90 fr |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.010 | 0.1511 | 0.1888 | 1.25× | 0.22403 | 0.22396 |
| 0.020 | 0.2380 | 0.2992 | 1.26× | 0.35431 | 0.35423 |
| 0.050 | 0.3826 | 0.4763 | 1.25× | 0.56814 | 0.56813 |
| 0.100 | 0.4847 | 0.6012 | 1.24× | 0.71218 | 0.71218 |

**`rolling_band_score` is window-length invariant to five decimals.** The entire
factor is carried by the illuminant and AWB channels, whose spectral-shape
measurements need bins to resolve against. In the same experiment
`peak_prominence` went 0.22 → 2.00 and `peak_to_median_ratio` 4.17 → 3.4 × 10⁶.

The factor therefore depends on the weights and **must be re-measured whenever
they move**: it is 1.40× at 0.30/0.50/0.20 and 1.25× at 0.10/0.65/0.25, because
shifting weight onto the invariant channel shrinks it. Weighting `rolling_band`
alone would remove the correction entirely and make a BurstFlicker-fitted
threshold transfer directly — a point in favour of that weighting beyond its rank
performance.

Applying ×1.25:

| operating point | BurstFlicker | production equivalent |
| --- | ---: | ---: |
| zero-FP | 0.1311 | **0.1639** |
| max Youden J | 0.0209 | **0.0261** |

### 2. Aggregation over windows — partially measured

A production video's score is the maximum over ~15 windows (5 minutes at 20 s
stride); a burst is a single window. Bootstrapping from the 369 clean
ground-truth scores, the median of max-of-15 is 0.0164 against 0.0005 for a
single window — a 32× shift in the *median*.

This bootstrap cannot speak to the tail: resampling with replacement only ever
returns observed values, so the max-of-15 p99 is pinned at the observed maximum
by construction. Treat the median shift as real and the tail as unmeasured.

### 3. Content — not corrected for

Fisheye egocentric framing keeps ceiling luminaires in frame almost continuously
and head rotation sweeps them through the periphery, producing quasi-periodic
global luminance change that is scene content rather than flicker. See the
fisheye section of `methodology.md`. Nothing here corrects for it.

### What the domain gap looks like

Clean against clean, at weights 0.30/0.50/0.20:

| | `flicker_score` | `illuminant` | `rolling_band` | `awb` |
| --- | ---: | ---: | ---: | ---: |
| VisionLabs clean (label `none`) | 0.264 | 0.252 | 0.242 | 0.338 |
| BurstFlicker clean (`gt`) | 0.003 | 0.002 | 0.005 | 0.003 |

Every channel reads 48–126× higher on "clean" production footage. Since the
measured window-length factor is only 1.25–1.40× and `rolling_band` is invariant
to it, **most of that gap is not a measurement artifact.** It is either real
sub-visible flicker or motion-driven content.

Inverting the synthetic `rolling_band_score` curve above — which is
weight-independent, being a per-detector score — gives an implied luminance
modulation depth per class:

| class | `rolling_band` mean | implied depth | range |
| --- | ---: | ---: | --- |
| VisionLabs `none` | 0.242 | **~1.1%** | 0.2% – 2.3% |
| VisionLabs `mild` | 0.559 | ~4.9% | 3.1% – 10.0% |
| VisionLabs `extreme` | 0.760 | ~10.0% | 7.1% – 10.0% |
| BurstFlicker `gt` | 0.005 | ~0.02% | |

`configs/detector.yaml` records that "2% is around where flicker becomes
perceptible". The VisionLabs `none` class implies ~1.1% — **real but
sub-perceptible**, which is exactly what a human labelling by eye would mark as
clean. The three classes form a physically sensible ladder at roughly 1% / 5% /
10%, all of them real flicker of increasing depth, with the `none` boundary
sitting at the eye's perception threshold rather than at zero.

One suggestive coincidence: the production-scaled zero-FP point is 0.1639 and the
lowest-scoring VisionLabs `none` video is 0.1644. With n=7 that is not
conclusive, but it is consistent with production "clean" beginning exactly where
BurstFlicker's clean ceiling ends — and therefore with the reference videos at
0.22–0.32 carrying real flicker the eye-labels missed.

Also worth recording, from `reports/discrimination_report.md`: on the production
corpus `illuminant_score` is flat across classes (0.252 / 0.224 / 0.257,
Spearman +0.034) and `awb_score` is inverted (0.338 / 0.289 / 0.309, Spearman
−0.184). Together they contribute `0.30 × 0.252 + 0.20 × 0.338 = 0.143` to a
clean video — over half its 0.264 — while ranking nothing. The weighted sum
reproduces the score exactly, as designed. That offset is a large part of why the
production clean floor sits at 0.164 rather than near zero.

## Recommendation

**Drop `mild_threshold`. Not to 0.15 — that is both higher than necessary for
safety and lower-yield than the alternatives.**

All rows below are round, configurable values with their exact measured rates —
not interpolations:

| if the constraint is | set `mild_threshold` to | FP rate | recall on ground truth |
| --- | ---: | ---: | ---: |
| zero false positives, non-negotiable | 0.13 | 0.0% | 61.5% |
| ≤ 1% false positives | 0.08 | 0.8% | 68.3% |
| ≤ 3% false positives | 0.03 | 2.7% | 85.6% |
| **≤ 5% false positives (preferred)** | **0.02** | **4.6%** | **90.2%** |

0.02 sits essentially on the max-Youden-J point (0.020880, FP 3.8%, recall 90.0%)
and is the one to pick absent a specific false-positive budget. Scaled for a 3 s
production window it becomes ≈ 0.025; the zero-FP choice becomes ≈ 0.164.

Leave `extreme_threshold` where it is. Nothing here bears on it.

## Limitations

1. **BurstFlicker-G flicker is generated, not captured.** Its flicker model may
   not match real rolling-shutter mains flicker in every respect.
2. **The flicker class has no severity grades**, so recall counts a barely-there
   flicker the same as an obvious one. The `band_depth` cross-check above is a
   partial substitute, not a replacement for graded labels.
3. **Only one of three domain-transfer effects is corrected.** Window length is
   measured and applied (×1.25); max-over-windows is measured only at the median;
   content and motion are not corrected at all.
4. **Recall is capped at 96.2%** by 14 clips the detector scores at zero despite
   strong measured banding. This is a detector defect and should be chased
   separately.
5. **Both corpora are narrow.** BurstFlicker-G is one capture rig; the VisionLabs
   set is 25 videos from one factory and two operators.
6. The `analysis_region` defaults applied here are fisheye policy (bottom 25%
   excluded) applied to non-fisheye footage. `--region no-bottom-exclusion` and
   `--region full-frame` exist to test that choice; the committed policy is the
   default so runs stay comparable with production.
7. **`docs/methodology.md` results sections are stale.** They describe the
   pre-rewrite broadband rolling-band detector (`flicker_score` Spearman −0.247,
   "the headline score is inverted"). The current fitted state is +0.910 with
   AUC 1.000. Do not read that document's measured-agreement tables as current.

## Reproducing

```bash
# Clean-side audit: one row per ground-truth frame
python scripts/score_gt_frames.py
#   -> output/burstflicker/gt_frame_scores.csv

# Both classes, plus the operating curve and candidate thresholds
python scripts/fit_none_threshold.py
#   -> output/burstflicker/pair_scores.csv

# Re-report instantly from saved scores (no 15-minute rescore)
python scripts/fit_none_threshold.py --from-scores --probe 0.02

# Compare a candidate weighting without editing the config
python scripts/fit_none_threshold.py --weights 0.0,1.0,0.0
```

Both scripts read weights and normalization scales from
`configs/detector.yaml`, so they measure the calibration as committed. Every
output row records `weight_illuminant` / `weight_rolling_band` / `weight_awb` and
`region_policy`, so two runs cannot be compared across different calibrations by
accident.

`WINDOW_LENGTH_FACTOR` in `scripts/fit_none_threshold.py` is currently 1.25 and
is only valid for weights 0.10/0.65/0.25. Re-measure it when the weights move.

### A note on machine sizing

BurstFlicker-G is 6960×4640, and PyAV decodes at source resolution before
reformatting to the 320×180 that gets measured — 97 MB per RGB frame. Eight
workers of that on an 8 GB machine exhausts memory and the process pool
deadlocks rather than failing cleanly. Both scripts now size the pool from a
probed clip and log the cap (`8 -> 4` on this machine).
