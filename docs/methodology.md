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

The rolling-band detector vertically smooths each row-profile matrix, removes
per-frame global brightness, computes vertical gradients, and tracks the
strongest edge across frames. It measures edge energy, band strength, vertical
velocity, position variance, and edge-energy periodicity. Unlike global AE
change, a true rolling band should provide both horizontal structure and
coherent vertical movement.

The AWB detector computes Lab a*/b* chroma magnitude per frame and runs Welch PSD
on the chroma signal to detect periodic color oscillation. It checks luma-chroma
decorrelation (since AWB hunting produces chroma changes independent of luma,
unlike illuminant flicker). It also detects AE hunting via low-frequency (<5 Hz)
luma oscillation that falls below mains frequencies. The final AWB score is
`max(adjusted_chroma_score, ae_score)`.

Detector measurements are normalized in a calibration layer, weighted, and
aggregated per window. A video's final score is its maximum window score rather
than its mean: short severe artifacts should not be diluted by otherwise clean
footage. The output retains the worst window for review.

## Distinguishing content and artifacts

Global spectral evidence alone can be caused by repetitive scene motion or
exposure control. The rolling-band measurement adds spatial constraints: it
requires persistent horizontal edge structure and motion primarily in the row
axis. Detrending reduces false positives from monotonic AE ramps. The AWB
detector adds chroma oscillation evidence, which is intentionally kept
separate from luminance flicker.

## Failure modes

Fast horizontal scene edges moving vertically can resemble a band, especially
when the scene is a conveyor or a striped surface. Rapid cuts, strobing content,
and periodic camera motion may cause spectral false positives. Very short or
intermittent artifacts can fall between windows; long windows improve coverage
but increase decode cost. Downsampling can weaken very thin bands. A static
horizontal lighting gradient may create edge energy but should have low tracked
velocity. AE/AWB hunting is explicitly handled and included in the final score via
the AWB detector.

## Scale and cost

The workload is independent per video and is parallelized across manifest entries
using `ProcessPoolExecutor`. At the configured cadence, the decoder analyzes
roughly 15% of each video's duration before the end-window adjustment, at
320×180. After manifest-batch processing, the system prints throughput metrics and
an estimated wall-clock time projection for the full 21.7K-hour corpus at various
worker counts. Before corpus-wide use, benchmark a representative shard to record
videos/hour, CPU hours/video-hour, network egress, and failure rate; multiply
measured CPU hours/video-hour by 21,700 corpus hours and the selected worker price.
The resumable manifest writer retains errors for controlled retry after URL refreshes.
