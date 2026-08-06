# Methodology and Failure Analysis

## Method

The system samples three-second windows every 20 seconds, always including an
end-anchored final window. Frames are decoded with PyAV and downsampled to
320×180 before analysis. This reduces spatial work substantially while retaining
the temporal samples required for flicker analysis. The three-second window
preserves the frame-rate-dependent Nyquist limit; mains-driven behavior can
appear as an alias below that limit, so the illuminant detector does not require
an observed peak to equal 50, 100, or 120 Hz exactly.

Each window yields global YUV luminance, Lab a*/b* chroma, and a luminance profile
for every image row. Luminance is linearly detrended and standardized, then
evaluated using a Hann-windowed Welch PSD. The illuminant measurement is the
strongest non-DC peak's prominence and peak-to-median PSD ratio.

Peak selection is bounded below, and that bound matters more than it sounds.
Selection was originally unbounded, and over a window long enough to resolve the
flicker band, egocentric head motion and exposure drift deposit far more energy
below 2 Hz than the flicker itself carries. On the reference flickering video the
chosen peak was 0.67 Hz of head motion while the real flicker sat at 9.99 Hz, so
every frequency-domain measurement described the wrong phenomenon — and because
the same estimate decided AE band membership, nearly every video was routed into
the AE-hunting channel. Restricting selection to 2 Hz and above recovers 9.99 Hz
at the unchanged 3-second window, and band coherence rises from 0.223 to 0.990.

That 9.99 Hz is what theory predicts, which is the strongest available evidence
that the detector sees genuine mains-driven flicker: 50 Hz Indian mains produces
100 Hz full-wave illuminant flicker, which at 29.97 fps aliases to
|100 − 3×29.97| = 10.09 Hz, within one 0.33 Hz spectral bin of the measurement.

The floor is therefore region-specific and is a known limitation. At 60 Hz mains,
120 Hz flicker aliases to |120 − 4×29.97| = 0.12 Hz, which falls *below* the floor
and would be discarded as motion. Sixty-hertz footage needs either a different
floor or a frame rate chosen so the alias lands clear of the motion band.

Spatial resolution, by contrast, is irrelevant to any of this: native 1280×720 and
the analysed 320×180 agree on band amplitude to three decimals, confirming that
aggressive spatial downsampling is close to free while temporal density is not.

Standardizing before the PSD is what makes those two measurements robust, but it
also discards amplitude: a 0.1% ripple and a 30% pulse at the same frequency
produce an identical spectrum shape. Spectral peakiness therefore answers only
"is this periodic?". Severity is carried separately by **modulation depth** — the
coefficient of variation of the detrended luminance, which is scale-free so dim
and bright scenes stay comparable. The two are *multiplied*, not averaged,
because flicker requires both: a clean spectral peak with a negligible swing is
invisible, and a large aperiodic swing is camera motion.

Detectors return raw measurements and no score. Converting a measurement into
bounded evidence happens in the calibration layer, so every tunable scale lives
in one place and can be fitted against the reference labels rather than being
hardcoded in a detector. Each scale is expressed as the measurement that scores
0.5, and normalization uses a soft saturation (`v / (v + reference)`) rather than
a clamped linear ramp — clamping collapses every strong artifact onto an
identical score and erases the differences a severity band needs.

The rolling-band detector vertically smooths each row-profile matrix, removes
per-frame global brightness, computes vertical gradients, and tracks the
strongest edge across frames. It measures edge energy, band strength, vertical
velocity, position variance, and edge-energy periodicity. Unlike global AE
change, a true rolling band should provide both horizontal structure and
coherent vertical movement.

The AWB detector computes Lab a*/b* chroma magnitude per frame and runs Welch PSD
on the chroma signal to detect periodic color oscillation. It measures luma-chroma
decorrelation (since AWB hunting produces chroma changes independent of luma,
unlike illuminant flicker), and how much chroma actually moves, applied as a soft
gate so a nearly colourless scene attenuates smoothly instead of cutting off at a
boundary. It also measures AE hunting from low-frequency luma oscillation. The
5 Hz limit separating the two is a *mechanism* boundary, not a severity
threshold: AE controllers hunt at a few hertz while the mains beat sits higher
and belongs to the illuminant detector. The two channels combine with `max`,
since they are alternative mechanisms and a video exhibiting one strongly should
not be discounted for lacking the other.

Normalized detector evidence is weighted and aggregated per window. A video's
final score is its maximum window score rather than its mean: short severe
artifacts should not be diluted by otherwise clean footage. The reported
per-detector scores are those of that same worst window, so the weighted sum of
the output columns reproduces `flicker_score` exactly and the row explains the
routing decision it carries. Reporting each detector's maximum independently
would mix evidence from different moments and reconcile with nothing.

## Distinguishing content and artifacts

Global spectral evidence alone can be caused by repetitive scene motion or
exposure control. The rolling-band measurement adds spatial constraints: it
requires persistent horizontal edge structure and motion primarily in the row
axis. Detrending reduces false positives from monotonic AE ramps. The AWB
detector adds chroma oscillation evidence, which is intentionally kept
separate from luminance flicker.

## Failure modes

## Measured agreement with the reference labels

All 25 colour-coded reference videos scored without error. Rank agreement with
the ordinal label, and the area under the ROC curve separating `extreme` from
`none` (0.5 is chance, n=25 — 7 none, 10 mild, 8 extreme):

| signal | Spearman | AUC extreme vs none | mean none / mild / extreme |
| --- | ---: | ---: | --- |
| `flicker_score` | **−0.247** | **0.286** | 0.590 / 0.559 / 0.566 |
| `horizontal_coherence` | **+0.519** | **0.911** | 0.365 / 0.543 / 0.800 |
| `illuminant_score` | +0.300 | 0.750 | 0.243 / 0.225 / 0.289 |
| `rolling_band_score` | **−0.492** | **0.125** | 0.908 / 0.867 / 0.839 |
| `awb_score` | −0.112 | 0.411 | 0.314 / 0.292 / 0.299 |

**The headline score is inverted.** `flicker_score` ranks clean footage slightly
*above* extreme footage, so no threshold placed on it can work, and the fitted
calibration is duly near chance: held-out accuracy 0.500, macro-F1 0.333.

The cause is `rolling_band_score`, which carries 50% of the weight and is
strongly anti-correlated with severity at AUC 0.125. Its components — band
strength, vertical velocity, position variance — are measured broadband, so on
clean footage vigorous head motion produces strong row-profile gradients and an
edge tracker that jumps between frames, scoring high on all three. It is
functioning as a motion detector, and egocentric fisheye supplies abundant
motion. Its clamped linear normalization then compresses everything into
0.767–0.990, so the inversion arrives with high confidence and little spread.

**The strongest available discriminator is `horizontal_coherence`, at AUC 0.911,
and it contributes nothing to the score.** It was built as a self-check on the
band model, but because it is evaluated only at the flicker frequency it is the
one measurement here that isolates banding from motion, which is precisely the
separation the task requires. The reference video cited in the brief scores 0.994
on it. Reweighting the aggregate around this measurement, and making the
rolling-band detector frequency-selective in the same way or removing its weight,
is the clear next step and is not yet done.

The band boundaries in `configs/detector.yaml` remain the provisional
carry-overs. Fitting them is pointless while the input to the fit is inverted.

## Limits of the calibration itself

The reference set is 25 videos from one factory and two operators, so a 20%
split leaves four validation videos. The split is by key hash and is not
stratified, and it happened to place no `none` video in validation at all, which
makes macro-F1 over that class meaningless. Any reported metric here is
indicative at best. A stratified or grouped split — the delivery metadata gives
an `operator_id` that would prevent same-wearer leakage — and considerably more
labels are both needed before the numbers carry weight.

Fisheye egocentric framing adds its own false positives. A ~180 degree field of
view keeps ceiling luminaires in frame almost continuously, and head rotation
sweeps them through the periphery, producing quasi-periodic global luminance
change that is scene content rather than illuminant flicker. Peripheral angular
compression means a small head movement displaces edge pixels enormously while
the centre barely moves, so motion-derived measurements are dominated by the frame
edges. Excluding the wearer's body helps but does not address either effect.

Fast horizontal scene edges moving vertically can resemble a band, especially
when the scene is a conveyor or a striped surface. Rapid cuts, strobing content,
and periodic camera motion may cause spectral false positives. Very short or
intermittent artifacts can fall between windows; long windows improve coverage
but increase decode cost. Downsampling can weaken very thin bands. A static
horizontal lighting gradient may create edge energy but should have low tracked
velocity. AE/AWB hunting is explicitly handled and included in the final score via
the AWB detector.

## Fisheye egocentric framing

The corpus is raw, un-dewarped fisheye: a 960x720 image circle pillarboxed inside
a 1280x720 frame, so 25% of every frame is matte black. Signals are therefore
measured over a detected analysis region rather than the whole frame. Deadness is
judged on each pixel's maximum over a window, not its mean, so a dim scene is not
mistaken for a crop.

Masking matters unevenly. Modulation depth is a ratio, so dead pixels scale its
numerator and denominator alike and cancel exactly — measured invariant to within
0.1%. Absolute measurements do not cancel: chroma variance and band strength read
25% low against absolute reference scales. Because this corpus is one camera at
one site, that bias is a constant a fitted scale would absorb, which is precisely
why it was worth removing — otherwise the calibration silently encodes one
camera's crop and breaks when a second is added.

The lowest rows of a ~180 degree egocentric frame are the wearer's own torso and
hands: differently lit, self-shadowed, and moving with the camera without
parallax. Measured band contrast there is about 2.7x weaker than mid-frame, so
`analysis_region.exclude_bottom_fraction` drops them. The default of 0.25 is a
prior, not a fitted value.

Rolling-shutter banding survives the lens. Exposure is sequenced per sensor row,
after the optics have already projected the scene, so fisheye distortion bends
scene content while leaving the bands straight — verified directly: vertical
slices of the frame agree on per-row brightness change to a correlation of
0.94-0.97. `horizontal_coherence` reports that agreement per video, measured only
at the flicker frequency because a raw frame difference is dominated by head
motion, which under a fisheye differs wildly between the left and right edges.
A low value alongside real periodic flicker indicates the frame was dewarped,
rectilinearized, or electronically stabilized, any of which bends the bands and
invalidates the rolling-band model.

## Data access

Videos are read in place from S3 rather than downloaded. Work is addressed by
durable `s3://bucket/key` URI; a short-lived URL is signed per video inside the
worker that decodes it, so no expiring credential is ever persisted in a
manifest or carried across a long batch. FFmpeg reads the signed URL over HTTP
range requests with connection reuse enabled, which means windowed sampling
transfers only the byte ranges it analyzes — roughly the same fraction of each
object that it decodes, rather than all ~570 MB. Egress, not CPU, is therefore
the quantity to watch when projecting corpus cost.

A live prefix listing is streamed, so enumerating the corpus costs constant
memory and work begins before listing finishes. Snapshotting the listing to a
manifest pins a reproducible work set, since the bucket itself changes over time.

## Scale and cost

The workload is independent per video and is parallelized across manifest entries
using `ProcessPoolExecutor`, with a bounded number of videos in flight so peak
memory does not scale with corpus size. At the configured cadence, the decoder analyzes
roughly 15% of each video's duration before the end-window adjustment, at
320×180. After manifest-batch processing, the system prints throughput metrics and
an estimated wall-clock time projection for the full 21.7K-hour corpus at various
worker counts. Before corpus-wide use, benchmark a representative shard to record
videos/hour, CPU hours/video-hour, network egress, and failure rate; multiply
measured CPU hours/video-hour by 21,700 corpus hours and the selected worker price.
Running workers in the bucket's own region avoids cross-region egress entirely,
which at corpus scale dominates compute cost.

### Where the time actually goes

Per 3-second window of 320×180 analysed content, signal extraction costs 0.16 s
and every feature, detector, and the aggregation together cost 0.09 s. Decoding
the 3840×2880 source that window came from costs an order of magnitude more.
The detectors are about 1% of a window, so they are the wrong thing to optimise
and are deliberately left on the CPU: Welch on a 90-sample signal and the
1081×135 phase-grid search in `rolling_band` are far too small to survive
kernel-launch overhead on a GPU.

Decoding is the workload, and it is the part worth moving to hardware.
`decode.backend` selects NVDEC where the driver, codec, and resolution allow it
and degrades to software otherwise, because a batch that silently dropped videos
for want of a GPU would be worse than a slow one.

Measured end to end on 24 real corpus videos through the ordinary batch path, on
one 16-core machine with an A10G:

| run | wall | user CPU |
|---|---|---|
| `--decode-backend cpu --workers 16` | 415.6 s | 6173.5 s |
| `--decode-backend cuda --workers 16` | 230.8 s | 1936.7 s |
| `--decode-backend cuda --workers 32` | 211.1 s | 1954.9 s |

1.8× the throughput for 3.2× less CPU, with no fallbacks and no video changing
severity band or route. The worst `flicker_score` difference across all 24 was
0.00055.

The bottleneck moves rather than disappearing. During the 16-worker GPU run the
NVDEC engine sat pinned at 100% while the SMs idled near 40% and VRAM held 6 GB
of 23 GB — about 375 MB per worker. That is why doubling to 32 workers bought
only 9%: the decode engine was already saturated, and past that point more
workers buy contention rather than throughput. Sizing for NVDEC, not for cores
or memory, is what matters now.

`decode.gpu_resize` moves the downscale into NVDEC as well and cuts per-window
CPU further, but its scaler is not swscale's, and a 16× vertical reduction is
exactly where the banding evidence lives. On real footage that is not a free
change: of the first three corpus videos tried, one moved from 0.352 to 0.310,
crossing `mild_threshold` and turning `review` into `accept`. Enabling it means
re-fitting `decision` against GPU-decoded scores. Verify any such change first:

```bash
uv run python scripts/compare_decode_backends.py s3://bucket/key.mp4 \
  --config configs/detector_1.yaml
```

Credentials, region, and read permission are checked once before any worker
starts, so an expired token halts the batch rather than producing one error row
per video. Resuming keeps successful rows and retries the rest, which is the
right default when the dominant failures are transient: expired temporary
credentials, throttling, and dropped connections part-way through an object.
