# Flicker Artifact Detection

Detects illuminant flicker, rolling horizontal bands, and AWB/AE hunting in egocentric video.
The implementation uses windowed PyAV decoding, `320×180` downsampling, shared
signal extraction, Welch PSD, row-profile motion measurements, and parallel
manifest batch processing with `ProcessPoolExecutor`.

## Setup

```bash
uv sync --locked
```

## Data access

Videos are read directly from S3. The corpus lives under
`s3://humyn-data-partners-prod/visionlab/visionlab/outbound/`, configured as
`s3.bucket` and `s3.prefix` in `configs/detector.yaml`.

Credentials come from the standard AWS chain — environment variables, a named
profile (`--aws-profile`), or an instance role. Nothing is read from the repo:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."      # required for temporary ASIA... credentials
```

`s3.region` must match the bucket's region (`ap-south-1`), because SigV4
presigning is region-scoped and a mismatch yields URLs that fail with 403.

**No presigned URL is ever stored.** A manifest holds only `s3://bucket/key`
URIs; each worker signs a short-lived URL for its own video immediately before
decoding it, and signatures are redacted from logs. FFmpeg then reads that
object over HTTP range requests, so windowed sampling transfers only the bytes
it analyzes rather than the whole file.

## Process the corpus from S3

List and score every video under the configured prefix:

```bash
uv run python main.py --s3-prefix \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

`--s3-prefix` also accepts a full URI (`s3://bucket/some/prefix/`) or a bare
prefix to read from the configured bucket. The listing is streamed and only a
bounded number of videos are in flight, so memory does not scale with corpus
size.

## Pin a run to a manifest

A live listing changes as the bucket does. Snapshot it once for a reproducible
work set, then score the snapshot:

```bash
uv run python main.py --s3-prefix --write-manifest data/manifest.csv
uv run python main.py --manifest data/manifest.csv \
  --output output/flag_manifest.csv --workers 8 --resume
```

The manifest schema is `key,source_uri,size_bytes,last_modified` and contains no
credentials. A newline-delimited `.txt` file of S3 URIs or keys works too. The
committed `data/manifest.csv` is such a snapshot of the 3,498-video reference
corpus.

## Process one video

```bash
uv run python main.py s3://humyn-data-partners-prod/visionlab/visionlab/outbound/India_Ahmedabad_Mirana_AssemblyLine_004_998.mp4
uv run python main.py path/to/local.mp4 --output output/flag_manifest.csv
```

## Hardware decode

Decoding is the entire cost of this pipeline: the sources are 3840×2880 at 29.97
fps while the analysed signal is 320×180, so per 3-second window software decode
spends 4.08 s of CPU against 0.24 s for every detector and feature combined.
NVDEC does the same window for 0.54 s.

`decode.backend` defaults to `auto`, which uses NVDEC when the driver, codec, and
resolution allow and falls back to software otherwise — an unsupported profile or
an exhausted GPU slows a batch down, it never drops videos from it. The chosen
backend is logged per video and can be forced from the command line:

```bash
uv run python main.py --s3-prefix --decode-backend cuda --workers 8 --resume
```

`decode.gpu_resize` decides whether NVDEC also performs the downscale. That is
the difference between 2.24 s and 0.54 s of CPU per window, but it substitutes
NVDEC's scaler for swscale's. Since `decision.mild_threshold` and
`decision.extreme_threshold` are fitted against measured scores, confirm the
backends agree on real footage before switching a batch:

```bash
uv run python scripts/compare_decode_backends.py \
  s3://humyn-data-partners-prod/.../clip.mp4 --config configs/detector_1.yaml
```

The script scores each video on both backends, refuses to fall back silently,
and exits non-zero if any video changes severity band or route.

**Worker count changes with the backend.** Under NVDEC the bottleneck is no
longer CPU cores but the GPU's decode engines, its memory (each worker holds its
own CUDA context), and S3 egress. A `--workers` value tuned for software decode
will be wrong; re-measure it.

## Resuming

`--workers N` sets the number of parallel process workers. `--resume` keeps
videos already scored successfully and retries everything else, including rows
that previously failed — the expected failures here are transient (expired
credentials, throttling, dropped connections). Failures are written as rows with
`status=error` so the output stays auditable, and each run rewrites the manifest
to exactly one row per `video_key`.

Batches fail fast: credentials, region, and read permission are checked with a
single request before any worker starts, so an expired token stops the run
(exit code 2) instead of producing one error row per video.

## Output schema

`video_key` is the S3 object key, so output rows join back to the bucket and to
the input manifest without ambiguity.

`illuminant_score`, `rolling_band_score`, and `awb_score` are measured at the
worst window — the same window as `worst_segment_start`/`_end` — so their
weighted sum reproduces `flicker_score` exactly and a row explains its own
routing decision. Weights live in `aggregation.weights`.

Two columns are diagnostics rather than evidence, and do not enter the score:

- `valid_fraction` — the share of the frame that carried live pixels. These are
  fisheye captures pillarboxed inside a 16:9 frame, so expect ~0.57 with the
  default bottom exclusion. A sharp departure means the framing changed.
- `horizontal_coherence` — how well vertical slices of the frame agree on the
  banding, measured at the flicker frequency. Rolling-shutter bands are imposed
  per sensor row, after the lens, so they stay straight under fisheye distortion.
  A low value alongside real flicker suggests the video was dewarped or
  stabilized, which invalidates the rolling-band model. **Currently uncalibrated
  — see the window-length defect in [methodology.md](docs/methodology.md) before
  relying on it.**

The CSV schema is fixed at version `1.0` and includes `video_key`, normalized
flicker score, severity band, accept/review/reject route, confidence margin,
worst-window timestamps, detector scores (`illuminant_score`, `rolling_band_score`, and `awb_score`),
processing time, and detector version. See `main.py:OUTPUT_FIELDS` for the authoritative field order.

## Calibration and evaluation

The reference labels exist only as cell fill colours in the delivery
spreadsheet, which no CSV export preserves. Extract them first — red is
`extreme`, orange is `mild`, green is `none`:

```bash
uv run python scripts/extract_labels.py \
  data/visionlabs_500h_delivery.xlsx \
  --output data/reference_labels.csv
```

Unrecognised colours are ignored rather than mapped to the nearest label, and an
uncoloured row yields no label at all rather than a clean verdict.

`reference_labels.csv` doubles as a manifest, so scoring the reference set and
calibrating against it are two commands:

```bash
uv run python main.py --manifest data/reference_labels.csv \
  --output output/reference_flag_manifest.csv --workers 8

uv run python main.py \
  --output output/reference_flag_manifest.csv \
  --calibrate-labels data/reference_labels.csv \
  --calibration-report reports/calibration_report.md
```

Thresholds are fitted on a deterministic 80% key-hash split, with held-out
metrics and a confusion matrix in the report.

## Tests

The suite is offline — S3 is stubbed, so no credentials are needed:

```bash
uv run python -m unittest discover -s tests -v
```

To verify live S3 access and scoring end to end, cheapest check first:

```bash
make verify-s3                              # or: ./scripts/verify_s3.sh --aws-profile NAME
```

It walks credentials → list permission → read permission → single-video decode →
parallel batch → resume, and stops at the first failure so you can tell which
layer is broken.

See [methodology.md](docs/methodology.md) for method, limitations, and scale
considerations.
