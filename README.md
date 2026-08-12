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

The corpus is normally named by a link sheet — see
[the next section](#process-the-corpus-named-by-the-link-sheet), which is the
primary input. Prefix enumeration remains for discovering objects the sheet does
not name (4,926 of them at present). List and score every video under the
configured prefix:

```bash
uv run python main.py --s3-prefix \
  --output output/window_metrics.jsonl \
  --window-dir output/window_metrics \
  --video-csv output/flag_manifest.csv \
  --workers 8 \
  --resume
```

`--s3-prefix` also accepts a full URI (`s3://bucket/some/prefix/`) or a bare
prefix to read from the configured bucket. The listing is streamed and only a
bounded number of videos are in flight, so memory does not scale with corpus
size.

## Process the corpus named by the link sheet

The work set is a spreadsheet, not a set of bucket prefixes. Each row's
`video_s3_link` is a key under `s3.bucket`, so the corpus is a fixed, reviewable
list rather than whatever a prefix happens to contain at listing time:

```bash
uv run python main.py --links input/all_s3links_updated.xlsx \
  --config configs/detector_1.yaml \
  --output output/window_metrics.jsonl \
  --workers 8 --resume
```

`--links` with no value reads `s3.links` from the config. `project_name`,
`video_id`, and `duration_s` are carried from the sheet into every output record,
so window scores group by project without joining back to the spreadsheet, and
the sheet's duration can be checked against the one the decoder reports.

### Link casing is reconciled against the bucket

**The sheet's links are lowercased and S3 keys are case-sensitive, so joining
bucket and column verbatim 404s on every row.** Measured against the live bucket,
all 140,519 of them:

| | |
|---|---|
| sheet | `raw/delhi_zetwork/2026-06-26/el0706/gx010086.mp4` |
| real | `raw/Delhi_ZetWork/2026-06-26/EL0706/GX010086.MP4` |

The bucket is therefore listed once per run (~150 requests, seconds) to build a
case-folded index of its 145,305 keys and recover the real casing. Against the
current sheet:

- **140,335 of 140,519 (99.87%)** resolve to exactly one real object
- **180** match nothing at any casing — the sheet names videos the bucket does
  not hold
- **4** are ambiguous: two real objects differ only in case, as
  `mirana_GoPro_IMU_SD21_GX040004` and `mirana_GoPro_IMU_sd21_GX040004` do. A
  lowercased link cannot pick between them and the pipeline will not pick for it,
  because scoring an arbitrary one attaches a verdict to the wrong video.

The 184 unresolvable rows are skipped and every one of them is written to
`reports/link_resolution.md` with its reason. They are skipped rather than
recorded as errors because they are not videos that exist: an error record would
only be a 404 that `--resume` retries forever.

Separately, 4,926 real `.mp4` objects in the bucket are **not** named by the
sheet. The sheet defines the work set, so those are out of scope — use
`--s3-prefix` if you want to sweep them too.

To avoid re-listing on every run, pin the resolved keys once and score the
snapshot:

```bash
uv run python main.py --links --write-manifest data/manifest.csv \
  --config configs/detector_1.yaml
uv run python main.py --manifest data/manifest.csv \
  --output output/window_metrics.jsonl --workers 8 --resume
```

The pinned manifest carries the real keys plus the sheet's `project_name`,
`video_id`, and `duration_s`, so it is equivalent to reading the sheet. Pass
`--no-resolve-case` to skip the listing entirely once the sheet carries exact
keys — against the current sheet it makes every row 404.

## Pin a run to a manifest

A live listing changes as the bucket does. Snapshot it once for a reproducible
work set, then score the snapshot:

```bash
uv run python main.py --s3-prefix --write-manifest data/manifest.csv
uv run python main.py --manifest data/manifest.csv \
  --output output/window_metrics.jsonl --workers 8 --resume
```

The manifest schema is `key,source_uri,size_bytes,last_modified` and contains no
credentials. A newline-delimited `.txt` file of S3 URIs or keys works too. The
committed `data/manifest.csv` is such a snapshot of the 3,498-video reference
corpus.

## Process one video

```bash
uv run python main.py s3://humyn-data-partners-prod/visionlab/visionlab/outbound/India_Ahmedabad_Mirana_AssemblyLine_004_998.mp4
uv run python main.py path/to/local.mp4 --output output/window_metrics.jsonl
```

A single video prints its whole window table, so the reported score can be
located in the timeline rather than taken on trust:

```
  11.2s at 29.97 fps, 1920x1080, decode=cuda
  flicker_score=0.5973 (max of 4 windows, mean 0.4923) band=extreme route=reject
  windows: 0 none, 3 mild, 1 extreme, 2 above positive_threshold

    #     start       end    score  band      illum  band_r     awb    freq   valid
  --------------------------------------------------------------------------------
    0      0.00      3.00   0.4297  mild      0.385   0.479   0.320   10.32   0.750
    1      3.00      6.00   0.5075  mild      0.438   0.550   0.425   10.32   0.750
    2      6.00      9.00   0.4344  mild      0.380   0.446   0.425    3.33   0.750
    3      8.24     11.24   0.5973  extreme   0.245   0.755   0.327    3.70   0.750   <- worst
```

Note windows 2 and 3 overlap: 11.24 s does not divide into whole 3 s windows, so
the end-anchored final window covers the 2.24 s remainder rather than leaving it
unmeasured. Every window stays exactly `window_duration` long, which is what
keeps their scores comparable.

## Hardware decode

Decoding is the entire cost of this pipeline: the sources are 3840×2880 at 29.97
fps while the analysed signal is 320×180, so a window spends an order of
magnitude more CPU being decoded than being measured. `decode.backend` defaults
to `auto`, which uses NVDEC when the driver, codec, and resolution allow and
falls back to software otherwise — an unsupported profile or an exhausted GPU
slows a batch down, it never drops videos from it. The backend actually used is
logged per video, and can be forced:

```bash
uv run python main.py --s3-prefix --decode-backend cuda --workers 16 --resume
```

Measured on 24 real corpus videos through this path, on a 16-core machine with
an A10G:

| run | wall | user CPU |
|---|---|---|
| `--decode-backend cpu --workers 16` | 415.6 s | 6173.5 s |
| `--decode-backend cuda --workers 16` | 230.8 s | 1936.7 s |
| `--decode-backend cuda --workers 32` | 211.1 s | 1954.9 s |

1.8× the throughput for 3.2× less CPU, no fallbacks, and no video changing
severity band or route (worst `flicker_score` difference 0.00055).

### Running without a GPU

**A GPU is never required.** `decode.backend: auto` probes the stream and falls
back to software whenever hardware decode is unusable — no CUDA device, no
`h264_cuvid` in the FFmpeg build, an unsupported codec or profile, exhausted GPU
memory, or one NVDEC session too many. Each reason is logged once per process and
the batch continues:

```
WARNING | Falling back to software decode: no CUDA hardware device is available
```

`--decode-backend cuda` also falls back rather than failing, so the same command
line works on both instance types. The only way to make a missing GPU fatal is
`decode.allow_fallback: false`, which exists precisely to prove the hardware path
is being taken in a benchmark — never for production.

Scores are effectively unchanged, though not bit-identical: NVDEC and libavcodec
implement the same normative H.264 reconstruction, and the downscale still happens
in swscale either way (`decode.gpu_resize: false`), so the two differ only by
rounding. On the 521 s 4K video measured below the gap was 8.5e-9
(`0.6249004788` on CPU against `0.6249089525` on NVDEC), and across 24 corpus
videos the worst difference was 0.00055 with no video changing band or route.

**Per-video wall time is not the reason to want a GPU** — throughput per machine
is. Same 521 s 4K video, 174 contiguous windows, on a 16-core box with an A10G:

| backend | wall | CPU time |
|---|---|---|
| `cpu` | 212.1 s | 1,264 s |
| `cuda` | 234.9 s | — |

Software decode is *faster* for one video, because frame threading spreads it
across ~6 cores while NVDEC has one decode engine. The GPU wins on a batch, where
that inverts: software decode costs 2.42 CPU-seconds per second of video, so a
16-core instance saturates at ~6.6× realtime no matter how many workers you start,
while NVDEC workers cost ~1 core each and scale until the decode engine or S3
egress runs out.

So a CPU-only fleet is entirely viable — size it by cores, not by `--workers`, and
budget roughly 4,300 machine-hours of a 16-core instance for the 28,216-hour
corpus.

### Where the time goes, and what actually helps

Measured per window on 4K corpus footage (A10G, 16 cores, contiguous windows):

| phase | per window | share |
|---|---|---|
| decode | 1275 ms | **93.6%** |
| signal extraction | 47 ms | 3.4% |
| detectors | 11 ms | 0.8% |
| colour conversion + mask | 14 ms | 1.0% |
| features + aggregation | 1 ms | 0.1% |

Optimising the analysis is pointless — it is 6% of the work. Only decode matters,
and on this hardware **the NVDEC engine is the wall**: during a batch the decoder
sits at 100% while the SMs idle at 8% and VRAM holds 2 GB of 23 GB. Five workers
bought 2.5× a single worker, not 5×, so `--workers` past a handful buys nothing on
one GPU.

That makes one A10G box good for ~10.6× realtime, or ~2,650 hours (110 days) for
the 28,216-hour corpus. The levers, in order of leverage:

1. **More GPUs, more machines.** The only lever that scales linearly. Needs a way
   to divide the work set — see the `--shard` note below.
2. **Run CPU workers alongside the GPU ones.** A GPU-bound run leaves ~16 cores
   idle. Software decode adds ~6.6× realtime on its own, so a mixed fleet on the
   same box reaches ~17× — about 1.6× for hardware already paid for. Start a
   second process with `--decode-backend cpu` against the same `--output`; resume
   keeps them from duplicating work.
3. **`decode.gpu_resize: true`** — see below. Worth 1.93× to a single stream but
   only ~1.11× to a saturated batch, because it removes CPU swscale work that was
   not the constraint. Take it for CPU-bound and mixed fleets.

What does *not* help: decoding fewer frames. The flicker sits at a ~10 Hz alias of
the mains beat, and the 3 s window at 29.97 fps is already close to what resolves
it, so frame skipping moves the measurement rather than speeding it up.

### `gpu_resize` is now worth reconsidering

The warning below was written when `decision` was considered fixed. It no longer
is — contiguous windows changed `flicker_score` and the thresholds need re-fitting
regardless, so the objection to re-fitting for `gpu_resize` has largely gone.

Re-measured on the 5-video sample, `gpu_resize` false against true:

- video-level: max |Δ| **0.0034**, and **0 of 5** changed severity band or route
- per-window: median |Δ| 0.0014, p99 0.0105, and only 11 of 1041 windows move more
  than 0.01
- window band flips: **2 of 1041** (0.19%), both `mild` → `none`

One window moved 0.35, and it is worth understanding rather than averaging away:
its dominant frequency flipped from 9.99 Hz to 2.33 Hz, so peak selection chose a
different peak, and every frequency-locked measurement followed it
(`phase_linearity` 0.986 → 0.694). That is not a scaling error — it is a borderline
spectrum where two peaks are near-equal in prominence, and any small perturbation
tips it. A pre-existing fragility that per-window output has made visible.

Five videos is a small sample. Validate on the reference set before committing.

**Size workers for NVDEC, not for cores.** During the 16-worker GPU run the
decode engine was pinned at 100% while the SMs idled near 40% and VRAM held 6 GB
of 23 GB (~375 MB per worker). Doubling to 32 workers therefore bought only 9%.
Beyond saturation, more workers buy contention.

`decode.gpu_resize` moves the downscale into NVDEC too and cuts CPU further, but
its scaler is not swscale's. **It is not safe at the current thresholds:** of the
first three corpus videos tried, one moved from 0.352 to 0.310, crossing
`mild_threshold` and turning `review` into `accept`. Enabling it means re-fitting
`decision` against GPU-decoded scores. Check any change first:

```bash
uv run python scripts/compare_decode_backends.py \
  s3://bucket/prefix/clip.mp4 --config configs/detector_1.yaml
```

The script scores each video on both backends, refuses to fall back silently,
and exits non-zero if any video changes severity band or route.

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

## Running in production

`scripts/run_batch.sh` is the entry point. It pins the thread environment, picks a
worker count for the decode backend, refuses to start on expired credentials or
alongside an existing batch, names its outputs after the row range, and tees to a
log:

```bash
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...

scripts/run_batch.sh                                # whole link sheet
scripts/run_batch.sh --from-row 0 --to-row 14051    # one shard
```

Stop it with Ctrl-C in the pane, or `scripts/stop_batch.sh`. **Never `kill <pid>`:**
Python takes SIGTERM without unwinding, so the worker pool is never shut down and
every worker is reparented to init and keeps decoding. `stop_batch.sh` signals the
process *group* and sweeps orphans from earlier runs.

Check the pipeline before a long run — six checks, cheapest first, so the first
failure names the broken layer:

```bash
./scripts/verify_s3.sh
```

### Splitting the corpus across instances

`--from-row` and `--to-row` select an **inclusive, zero-based** range of the input
file's rows. Rows are sliced from the sheet *before* anything is resolved, so the
ranges are stable: rows that turn out to be unresolvable leave their instance with
less to do rather than shifting every later boundary.

Don't type the ranges — have them computed, because a hand-made off-by-one leaves
a silent gap in the corpus or scores videos twice. The shard count is yours to
choose; the ranges tile exactly for any N:

```bash
uv run python main.py --links --print-shards 20
```

```
140,519 rows in input/all_s3links_updated.xlsx -> 20 shards, balanced by duration
28,216 hours of video total

  shard   --from-row     --to-row       rows     hours
  ----------------------------------------------------
      0            0         6649      6,650    1,411h
      1         6650        12060      5,411    1,410h
      ...
     19       132135       140518      8,384    1,411h

  covers 140,519 of 140,519 rows, no overlap, no gap
```

Row counts differ on purpose — the split balances *hours of video*, since decode
dominates the cost. `--shard-by rows` gives equal row counts instead, at ~1.2×
imbalance in hours. `--shard-format tsv` prints `index<TAB>from<TAB>to` for a
launcher to read in a loop.

Run one shard per instance with `scripts/run_batch.sh --from-row A --to-row B`.
Each writes its own records and rollup, named for the range:

```
output/shards/rows_0000000-0014051.jsonl     # per instance
output/shards/rows_0000000-0014051.csv       # per instance
output/shards/rows_0000000-0014051.log       # per instance
output/window_metrics/raw/.../GX010085.csv   # shared, one file per video
```

**Why the split is what it is.** The JSONL and rollup CSV are opened with `w`, so
two instances sharing one would truncate each other — they must be per shard. The
per-video window CSVs are keyed by the video, so instances working different rows
can never collide and the tree assembles itself.

`--resume` is on by default in the script and works per shard, so an instance that
dies is restarted with the identical command.

To run several shards on one box deliberately, give each a distinct range and set
`ALLOW_CONCURRENT=1` — the guard against a second batch exists because an
accidental one looks like "the GPU is slow" from the outside.

Merge the shards when they finish; several JSONL files are read in order:

```bash
uv run python scripts/export_windows.py output/shards/*.jsonl \
  --videos output/flag_manifest.csv
```

### Tuning per instance

| variable | default | note |
|---|---|---|
| `WORKERS` | 5 on GPU, `2 × nproc` on CPU | NVDEC saturates near 3; software decode wants oversubscription because workers block on S3 ~18% of the time |
| `BACKEND` | `auto` | `cpu` on a GPU-less instance; `auto` already falls back |
| `CONFIG` | `configs/detector_1.yaml` | |
| `OUTPUT_ROOT` | `output` | |

A pinned manifest avoids every instance listing the bucket to recover link casing.
Snapshot once, then shard the manifest with the same flags:

```bash
uv run python main.py --links --write-manifest data/manifest.csv
scripts/run_batch.sh --manifest data/manifest.csv --from-row 0 --to-row 14051
```

## Publishing results to S3

Results are written locally first and copied up, so a failed or forbidden upload
costs the copy and never the run's work.

The destination lives in the config, so six instances do not each retype it:

```yaml
s3:
  output: s3://stage-egocentric-humyn-data/flicker_results/
  # output_region: ap-south-1   # only if it differs from s3.region
```

**Note which bucket is which.** The corpus is *read* from `prod-egocentric-humyn-data`
and results are *written* to `stage-egocentric-humyn-data`, because the EC2
instances carry stage credentials: they can write to stage and cannot write to
prod. `--s3-output` overrides the config for one run; `--no-s3-output` keeps a run
local only.

The local `output/` layout is reproduced verbatim beneath the prefix:

```
s3://stage-egocentric-humyn-data/flicker_results/
├── shards/
│   ├── rows_0000000-0023170.jsonl     one per instance
│   └── rows_0000000-0023170.csv
└── window_metrics/
    └── raw/Delhi_ZetWork/2026-06-20/DV0051/GX010085.csv
```

Six instances write six shard files into the same prefix and share the
`window_metrics/` tree, which is safe because those are keyed by video.

Per-video CSVs go up as each video finishes, so a long batch accumulates results
remotely as it runs. The JSONL and the rollup CSV are appended to all run and are
copied once at the end — re-uploading a 25 GB JSONL per video would cost more than
the batch. `scripts/publish_output.py` copies an existing tree up at any time:

```bash
uv run python scripts/publish_output.py output \
  s3://stage-egocentric-humyn-data/flicker_results/ --skip detector.log
```

### This bucket also holds the corpus

`prod-egocentric-humyn-data` holds the 140,335 source videos, so the publisher is
deliberately narrow. `src/data/publish.py` is the only module that writes to S3,
and it calls `put_object` and `upload_file` and nothing else — **there is no
delete, no copy, and no bucket-level call anywhere in it.** Two guards keep writes
inside the destination:

- a destination without a prefix is **refused**, so `s3://bucket` alone cannot
  scatter results across the bucket root;
- every key is rebuilt from the prefix and re-checked against it before the
  request, so a relative path containing `..` cannot climb out onto corpus data.

Write access is verified once before any decoding, for the same reason the reader
verifies read access: a permission problem should stop the run at the first
request, not after hours of work. The probe writes a small
`.flicker_write_check` object and leaves it — removing it would need delete
permission this tool never asks for.

### Required permission

The run stops immediately with the permission it needs if it is missing. Grant
`s3:PutObject` scoped to the prefix, not the bucket:

```json
{
  "Effect": "Allow",
  "Action": "s3:PutObject",
  "Resource": "arn:aws:s3:::stage-egocentric-humyn-data/flicker_results/*"
}
```

Until then, `--s3-output-dry-run` (or `--dry-run` on the script) resolves and logs
every destination key without issuing a request, which is the one mode a
read-only role can exercise end to end.

## Window coverage

Windows are **contiguous**: the stride equals the window length, so the schedule
tiles the video end to end. A ten-minute video is 200 windows of 3 s, and no
moment between the first and last window goes unmeasured. `sampling.stride` is
`null` to say so, rather than a `3.0` that could drift apart from
`window_duration` and silently reintroduce gaps.

The previous 20 s stride *sampled* the video, measuring 15.5% of it, so an
artifact could fall entirely between two windows. Contiguous coverage removes
that blind spot, and both sides of the trade are large. Measured on one 521 s 4K
corpus video, same footage, NVDEC, only the stride changed:

| | windows | coverage | wall | flicker_score | band | route |
|---|---|---|---|---|---|---|
| stride 20 s | 27 | 15.5% | 47.2 s | 0.5816 | mild | review |
| contiguous | 174 | 100% | 286.0 s | 0.6249 | **extreme** | **reject** |

Two things to take from that.

**Cost rises ~6×.** 0.426 s of worker time per second of video after the
one-pass decode below, and S3 egress goes from 15.5% to 100% of every object.
Across the 140,335 resolvable videos (28,216 hours, 498.5 TB):

| workers | contiguous | was (stride 20 s) |
|---|---|---|
| 8 | 1,503 h | 319 h |
| 32 | 376 h | 80 h |
| 64 | 188 h | 40 h |
| 128 | 94 h | 20 h |

Same-region S3 → EC2 transfer is free, so the 498 TB is a throughput concern
rather than a bill; it would not be from outside `ap-south-1`.

### A gapless schedule is decoded in one pass

Seeking to each window re-decodes from the preceding key frame, so the frames
between that key frame and the window start are reconstructed and thrown away —
~200 times for a ten-minute video. `VideoReader.read_windows` takes the whole
schedule instead, seeks once, and hands each frame to whichever windows span it.

Measured on the same 521 s 4K video, 174 windows, NVDEC:

| route | wall | frames decoded |
|---|---|---|
| seek per window | 276.1 s | 15,644 |
| one forward pass | 222.2 s | 15,644 |

×1.24, or 20% less wall time, for **bit-identical output** — same frame count and
same pixel checksum, so no score moves. It is not larger because the irreducible
part is decoding 100% of the frames; what was removed is only the re-decode from
each key frame.

The route is chosen from the schedule, not configured, so it cannot disagree with
the sampler: a gapless schedule takes the single pass, and a schedule with gaps
keeps the per-window seek, because walking the gaps would decode frames no window
ever asks for. Windows are yielded as they complete, so only the one or two
currently open are held — a 3 s window at 320×180 is ~15 MB, and 200 of them at
once would not fit in a worker's share of memory.

**`flicker_score` shifts upward, and routes change with it.** The score is a
maximum over windows, so taking it over 200 draws instead of 31 lands further up
the per-window distribution for identical footage — +0.0433 on the video above,
which was enough to cross `extreme_threshold` and turn `review` into `reject`.

> **`decision` needs re-fitting.** Both thresholds were fitted against
> 31-window maxima and are now applied to 200-window maxima, which biases the
> whole corpus toward `mild` and `extreme`. Re-run `--calibrate-labels` against
> contiguous scores before trusting the routes, and do not compare scores across
> the two settings — the 39,977 videos already in `output/flag_manifest.csv` were
> scored at stride 20 s and are not comparable to new rows.

A stride larger than the window returns to sampling if you want the old
behaviour: set `sampling.stride: 20.0`.

## Output schema

Detection runs per window and reports per window. A ten-minute video is 200
windows and produces 200 rows, not one. The rollup is still computed, but it is
now derived from records the consumer can see, so a score can always be located
in the timeline instead of being taken on trust.

Schema version is `2.0`. Three files are written, all from the same records:

| file | unit | what it is for |
|---|---|---|
| `--output output/window_metrics.jsonl` | one line per video | the durable output. Nested: rollup plus every window with its raw measurements. `--resume` reads this. |
| `--window-dir output/window_metrics/` | one CSV per video, 200 rows per 10 min | per-video window metrics, at the video's own key path |
| `--video-csv output/flag_manifest.csv` | one row per video | the rollup, in the original flag-manifest schema |

Why JSONL is the durable one and the CSVs are derived: a video is written in a
single atomic append, so `--resume` keeps video granularity and a batch killed
mid-run can never leave a video's windows half-written. A per-window CSV as the
primary output would give that up — one video's rows would interleave with other
workers' and a truncated file would leave rows that look complete.

Window metrics are **one CSV per video**, addressed exactly like the video itself:
the key's directories are reproduced under the root and only the extension
changes, so

```
raw/Delhi_ZetWork/2026-06-18/WRK-73637/GX060044.MP4
  -> output/window_metrics/raw/Delhi_ZetWork/2026-06-18/WRK-73637/GX060044.csv
```

One file per video means no two workers ever write the same file, a killed batch
leaves complete files rather than a truncated table, and a reviewer opens exactly
the video they are looking at. Concatenating the tree reproduces one combined
table; a failed video gets no file at all, because an empty CSV would claim it was
measured and found clean.

At 200 windows per ten-minute video that is ~28M rows across ~140K files for the
full corpus, so pass `--no-window-dir` on a corpus-scale run and materialise only
the slice you want afterwards — the JSONL can rebuild either shape:

```bash
# just the suspect windows, as one table
uv run python scripts/export_windows.py output/window_metrics.jsonl \
  --band mild extreme --windows output/suspect_windows.csv

# or rebuild the per-video tree
uv run python scripts/export_windows.py output/window_metrics.jsonl \
  --window-dir output/window_metrics
```

### The record

```json
{
  "schema_version": "2.0",
  "status": "ok",
  "error": "",
  "video_key": "raw/Delhi_ZetWork/2026-06-17/HAA344/GX010091.MP4",
  "detector_version": "0.2.0",
  "completed_at": "2026-08-11T10:13:59+00:00",
  "processing_time_seconds": 2.07,
  "source": {"duration": 11.24, "fps": 29.97, "width": 1920, "height": 1080,
             "decode_backend": "cuda"},
  "aggregate": {
    "flicker_score": 0.5973, "mean_score": 0.5135,
    "severity_band": "extreme", "route": "reject", "confidence": 0.0326,
    "total_windows": 2, "positive_windows": 1, "worst_window_index": 1,
    "worst_segment_start": 8.2446, "worst_segment_end": 11.2446,
    "band_counts": {"none": 0, "mild": 1, "extreme": 1},
    "scores": {"illuminant": 0.2447, "rolling_band": 0.7554, "awb": 0.3273},
    "horizontal_coherence": 0.9024, "valid_fraction": 0.75
  },
  "windows": [
    {
      "index": 1, "start_time": 8.2446, "end_time": 11.2446,
      "fps": 29.97, "frame_count": 90,
      "score": 0.5973, "severity_band": "extreme", "route": "reject",
      "confidence": 0.0326, "is_worst": true,
      "scores": {"illuminant": 0.2447, "rolling_band": 0.7554, "awb": 0.3273},
      "diagnostics": {"valid_fraction": 0.75, "horizontal_coherence": 0.9024},
      "measurements": {
        "illuminant": {"dominant_frequency": 3.70, "modulation_depth": 0.0111, ...},
        "rolling_band": {"band_amplitude": 0.1135, "phase_linearity": 0.99, ...},
        "awb": {"ae_dominant_frequency": 0.666, "ae_in_band": true, ...}
      }
    }
  ]
}
```

`measurements` is each detector's raw, pre-normalization output, read
generically off the detector metric dataclasses. It is what makes a window
score traceable without re-running the decode, and a detector that gains a
measurement reports it without a change to the writer.

### Invariants

- **The weighted detector scores reproduce the window's score, on every window.**
  `Σ wᵢ·sᵢ / Σ wᵢ` over `scores`, with weights from `aggregation.weights`.
  Taking each detector's maximum across windows mixed evidence from different
  moments, so the columns could not explain the routing they accompanied.
- **`aggregate.flicker_score` is the maximum over `windows[].score`**, and
  `worst_window_index` is that window's index. A maximum, not a mean: a
  three-second artifact in a ten-minute video is ~1/31 of the mean and would
  sail through as clean. Every column in the video row comes from that one
  window, so the row reconciles with itself.
- **A failed video has `windows: []`, never zero-filled windows.** No
  measurement was taken and a zero would read as one. `--resume` retries
  exactly these.

`video_key` is the S3 object key, so records join back to the bucket and to the
input manifest without ambiguity. The window table repeats the video's key,
project, score, band, and route on every row, so it stands alone in a dataframe.

`source.project_name`, `source.video_id`, and `source.sheet_duration` come from
the link sheet rather than the file. `sheet_duration` sits beside the decoded
`duration` instead of replacing it, so a sheet that disagrees with the container
is visible rather than reconciled away. `video_id` is carried but never used to
identify a video — it is not unique in this corpus (119,532 duplicates); the S3
key is.

Two fields are diagnostics rather than evidence, and do not enter the score.
They are now per window, which is where they were always measured:

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

See `src/core/records.py` for the authoritative `VIDEO_FIELDS` and
`WINDOW_FIELDS` column order.

### Window thresholds

`decision.window` grades each window in its own right, separately from
`decision.mild_threshold` / `extreme_threshold`, which grade the video. They are
separate because the two quantities are not drawn from the same distribution: a
video score is the maximum over 200 windows and therefore sits far up the
per-window distribution. Bootstrapping the clean ground-truth clips puts the
median of max-of-15 at 0.0164 against 0.0005 for a single window, and a maximum
over 200 draws sits further up still than the max-of-31 the thresholds were
fitted against.

**A boundary fitted against video maxima is therefore too high to apply to one
window, and window bands under it read conservatively.** The defaults are left
equal to the video thresholds on purpose — that keeps one documented number
rather than inventing a second, unfitted one. Fitting them needs labels on
windows, not on videos, and no such set exists yet. Until then, treat
`windows[].score` as the calibrated quantity and `windows[].severity_band` as
provisional.

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
  --output output/reference_window_metrics.jsonl \
  --video-csv output/reference_flag_manifest.csv --workers 8

uv run python main.py \
  --video-csv output/reference_flag_manifest.csv \
  --calibrate-labels data/reference_labels.csv \
  --calibration-report reports/calibration_report.md
```

Thresholds are fitted on a deterministic 80% key-hash split, with held-out
metrics and a confusion matrix in the report. Calibration reads the *video*
table, because the reference labels are per video; `decision.window` is
deliberately a separate pair of thresholds and cannot be fitted from them (see
[Window thresholds](#window-thresholds)).

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
