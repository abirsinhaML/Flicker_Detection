# Flicker Artifact Detection

Detects illuminant flicker, rolling horizontal bands, and AWB/AE hunting in egocentric video.
The implementation uses windowed PyAV decoding, `320×180` downsampling, shared
signal extraction, Welch PSD, row-profile motion measurements, and parallel
manifest batch processing with `ProcessPoolExecutor`.

## Setup

```bash
uv sync --locked
```

## Process one video

```bash
uv run python main.py path/to/video.mp4 --output output/flag_manifest.csv
```

## Process a manifest

```bash
uv run python main.py \
  --manifest data/manifest.csv \
  --output output/flag_manifest.csv \
  --workers 4 \
  --resume
```

`--workers N` specifies the number of parallel process workers (using `ProcessPoolExecutor`) for manifest-batch processing.
`--resume` preserves completed rows and retries only missing keys. Failures are
also emitted as rows with `status=error`, making the output auditable and safe
to restart after expired URLs or transient network failures.

## Output schema

The CSV schema is fixed at version `1.0` and includes `video_key`, normalized
flicker score, severity band, accept/review/reject route, confidence margin,
worst-window timestamps, detector scores (`illuminant_score`, `rolling_band_score`, and `awb_score`),
processing time, and detector version. See `main.py:OUTPUT_FIELDS` for the authoritative field order.

## Calibration and evaluation

Export the reference labels as a CSV with these columns:

```text
video_key,label
```

Use `none`, `mild`, and `extreme` labels (`orange` should be mapped to `mild`,
and `red` to `extreme`). Once the reference flag manifest exists, fit on a
deterministic 80% key split and write held-out metrics:

```bash
uv run python main.py \
  --output output/reference_flag_manifest.csv \
  --calibrate-labels data/reference_labels.csv \
  --calibration-report reports/calibration_report.md
```

The supplied Google Sheet exports the video manifest but not its cell colors,
so `reference_labels.csv` must be exported separately or provided by the team.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

See [methodology.md](docs/methodology.md) for method, limitations, and scale
considerations.
