# Running the flicker detector in Docker

Hand-off notes for whoever operates the fleet. One container per shard, each given
a row range of the link sheet. **The number of shards is your choice** — pick it
from the capacity you have, compute the ranges with one command, and the shards
will tile the corpus with no overlap and no gap whatever N you choose.

**Not yet built or run anywhere.** The Dockerfile is written against verified facts
about the dependencies, but no `docker build` has been executed — expect to iterate
once on the first build.

---

## What the job is

Score 140,519 videos (28,216 hours, read from S3) for flicker artifacts and write
per-window results back to S3.

- **Reads** `s3://prod-egocentric-humyn-data/raw/…` — the corpus
- **Writes** `s3://stage-egocentric-humyn-data/flicker_results/` — the results
- Also writes to a local volume, which is the source of truth; the S3 copy is
  derived from it
- Decode is ~94% of the cost, so this is GPU-bound on NVDEC where a GPU is present
  and CPU-bound otherwise. It runs correctly either way.

Rough sizing, from a measured ~10.6× realtime on a single-GPU box:

| shards | hours each | ~days each |
|---|---|---|
| 6 | 4,700 | 20 |
| 10 | 2,820 | 12 |
| 20 | 1,410 | 6 |
| 40 | 705 | 3 |

Any shard can be stopped and restarted at any time: `--resume` skips completed
videos.

---

## Host requirements

| | |
|---|---|
| Docker | any recent version |
| GPU (optional, ~6× faster) | NVIDIA driver + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then `--gpus all` |
| Disk | budget ~1.5 GB of results per 1,000 videos, on the mounted `output/` volume |
| RAM | ~500 MB per worker |
| Credentials | an **IAM instance profile**; see below |

Without a GPU it still works: the pipeline probes NVDEC, logs one line, and falls
back to software decode.

---

## Credentials

The container reads one bucket and writes another, so the identity it runs as needs
both:

```json
[
  { "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": ["arn:aws:s3:::prod-egocentric-humyn-data",
                 "arn:aws:s3:::prod-egocentric-humyn-data/*"] },
  { "Effect": "Allow",
    "Action": "s3:PutObject",
    "Resource": "arn:aws:s3:::stage-egocentric-humyn-data/flicker_results/*" }
]
```

Attach it as an **instance profile** and pass no credentials to the container —
boto3 finds them through the metadata service and refreshes them automatically.
Shards run for days and exported STS/SSO tokens expire in hours; when they do the
batch stops being able to read the corpus, not just to upload.

> **IMDSv2 hop limit — the one Docker-specific trap.** A container sits one network
> hop further from the metadata service than the host, and the default hop limit of
> 1 silently blocks it. The container then reports `Unable to locate credentials`
> *even with a role correctly attached*. Raise it per instance:
>
> ```bash
> aws ec2 modify-instance-metadata-options \
>   --instance-id i-xxxxxxxx \
>   --http-put-response-hop-limit 2 \
>   --http-tokens required
> ```

Static keys also work (`-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY
-e AWS_SESSION_TOKEN`) but only long-lived ones, for the expiry reason above.

---

## Build

```bash
docker build -t flicker:0.2.0 .
```

For a fleet, build once and push to ECR rather than building on every instance:

```bash
ACCOUNT=<aws-account-id>; REGION=ap-south-1; REPO=flicker
aws ecr create-repository --repository-name $REPO --region $REGION
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker tag flicker:0.2.0 $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0
```

Then on each instance:

```bash
IMAGE=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker pull $IMAGE
```

---

## Work out the shard ranges

Choose N, then ask the image. This needs no AWS credentials and touches nothing:

```bash
docker run --rm -v /data/input:/app/input:ro --entrypoint python \
  $IMAGE main.py --links --print-shards 20
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

Row counts differ on purpose: the split is balanced on **hours of video**, because
decode dominates the cost so hours predict runtime and row counts do not. Balanced
by row count instead, the heaviest shard carries ~1.2× the hours of the lightest
and that instance is still running when the rest have finished. Balanced this way,
20 shards land within one hour of each other out of 1,411.

Add `--shard-format tsv` for `index<TAB>from<TAB>to` and nothing else, so a
launcher can consume it directly — see below. `--shard-by rows` if you want equal
row counts instead.

---

## Run

Two volumes are required. The link sheet is not baked into the image, and results
must outlive the container.

**One shard on one instance**, substituting that instance's range:

```bash
mkdir -p /data/input /data/output
# copy input/all_s3links_updated.xlsx (3.8 MB, in the repo) to /data/input/

docker run -d \
  --name flicker \
  --restart unless-stopped \
  --gpus all \
  --init \
  -v /data/input:/app/input:ro \
  -v /data/output:/app/output \
  $IMAGE --from-row 0 --to-row 6649
```

The container's arguments are the row range and nothing else. Output paths, worker
count, thread pinning, `--resume` and the S3 destination are all handled inside.

**Several shards on one instance** — give each a distinct range, its own container
name, and set `ALLOW_CONCURRENT=1` to bypass the single-batch guard:

```bash
docker run -d --name flicker-0 -e ALLOW_CONCURRENT=1 ... $IMAGE --from-row 0 --to-row 6649
docker run -d --name flicker-1 -e ALLOW_CONCURRENT=1 ... $IMAGE --from-row 6650 --to-row 12060
```

They can share one `/data/output` volume: shard files are named after the range and
per-video results are keyed by video, so nothing collides.

**Launching N shards from a script**, so the ranges are never retyped:

```bash
N=20
IMAGE=<your image>
docker run --rm -v /data/input:/app/input:ro --entrypoint python \
  $IMAGE main.py --links --print-shards $N --shard-format tsv \
| while IFS=$'\\t' read -r shard from to; do
    docker run -d \
      --name "flicker-$shard" \
      --restart unless-stopped --gpus all --init \
      -e ALLOW_CONCURRENT=1 \
      -v /data/input:/app/input:ro \
      -v /data/output:/app/output \
      "$IMAGE" --from-row "$from" --to-row "$to"
  done
```

Run that once per instance with only its own subset of shards, or adapt it to your
orchestrator — the only contract is that every shard runs exactly once somewhere.

### Tuning

| env var | default | when to change |
|---|---|---|
| `WORKERS` | `2 × nproc` | With a GPU, NVDEC saturates near 3 concurrent streams: 32 workers mostly queue, each holding ~500 MB. Try 6 and compare the logged `Rate:`. **Set it explicitly if you limit the container with `--cpus`**, since `nproc` still reports the host's count. |
| `BACKEND` | `auto` | `cpu` to force software decode |
| `ALLOW_CONCURRENT` | `0` | `1` to run more than one container per host |
| `CONFIG` | `configs/detector_1.yaml` | |

---

## Monitor

```bash
docker logs -f flicker
docker exec flicker python scripts/check_progress.py
```

`check_progress.py` reports, per shard, how many videos are done out of the
resolvable total, the recent rate, and an ETA:

```
rows_0000000-0006649  rows 0..6649
  [##......................................]   5.2%
  scored     341 of 6,602 resolvable   (340 ok, 1 error)
  rate       98 videos/hour over the last 4.1h
  remaining  6,261 videos -> ~2.7d at this rate
```

On the first run confirm both of these in `docker logs`:

- `decode=cuda` on the metadata lines — otherwise it fell back to software and
  will use ~6× the CPU
- `Verified write access to s3://stage-egocentric-humyn-data/flicker_results/`

From the host, `nvidia-smi dmon` should show `dec` near 100%. If `dec` is low while
the CPU is pegged, set `decode.gpu_resize: true` in `configs/detector_1.yaml` — it
moves the downscale into NVDEC and removes the CPU-side work.

---

## Stop and restart

```bash
docker stop -t 60 flicker      # allow time to unwind the worker pool
docker start flicker           # resumes from the last completed video
```

`--resume` is always on. Granularity is **per video**, so anything mid-decode when
stopped is redone from the start — minutes, not hours. Give `docker stop` a
generous timeout; the default 10 s is often not enough to flush cleanly, though a
hard kill loses nothing already written.

`--restart unless-stopped` brings containers back after an instance reboot, and
they resume.

---

## Output

Results appear in the mounted volume and, in parallel, in S3:

```
/data/output/
├── shards/rows_0000000-0006649.jsonl    every window of every video (durable record)
├── shards/rows_0000000-0006649.csv      one row per video
├── shards/rows_0000000-0006649.log      the run log
└── window_metrics/raw/<key path>.csv    one CSV per video, at the video's own key

s3://stage-egocentric-humyn-data/flicker_results/
├── shards/rows_0000000-0006649.jsonl
├── shards/rows_0000000-0006649.csv
└── window_metrics/raw/<key path>.csv
```

**Per-video CSVs upload as each video finishes. The shard `.jsonl` and `.csv` upload
only when the batch finishes** — they are appended to all run, and re-sending a
multi-gigabyte file after every video would cost more than the batch. On a
multi-day shard that leaves the durable record un-copied for the duration, so sync
periodically from the host:

```cron
0 * * * * /usr/bin/flock -n /tmp/flicker_upload.lock \
  docker exec flicker python upload_output.py --skip-existing --skip detector.log \
  >> /var/log/flicker_upload.log 2>&1
```

`--skip-existing` compares object size and skips unchanged files, so after the
first pass it sends only what grew.

All shards write into the same S3 prefix and share `window_metrics/`, which is safe
because those files are keyed by video — no two shards can write the same one.

---

## Smoke test first

Before committing days of compute, run one container over ten rows. It finishes in
minutes and exercises credentials, GPU detection, S3 write access and both output
paths:

```bash
docker run --rm --gpus all \
  -v /data/input:/app/input:ro -v /data/output:/app/output \
  $IMAGE --from-row 0 --to-row 9
```

Expect roughly 4 of those 10 to score and 6 to be skipped as unresolvable — rows
0–9 sit in a part of the sheet that names objects the bucket no longer holds. That
is normal for this slice; corpus-wide the miss rate is 0.13%.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `Unable to locate credentials` | No instance profile, or the IMDSv2 hop limit is 1. See above. |
| `cannot write to s3://…` at startup | Missing `s3:PutObject` on the results prefix, or a region mismatch — the error names which. Nothing is decoded before this check passes. |
| `decode=cpu` with a GPU present | `--gpus all` missing, or `nvidia-container-toolkit` not installed on the host. Runs correctly, ~6× the CPU. |
| `refusing to start: N main.py process(es) already running` | Another batch is alive in that container. `docker exec flicker scripts/stop_batch.sh`, or use `ALLOW_CONCURRENT=1` if it is deliberate. |
| `S3 credentials are missing or expired` | The preflight refused to start. Expected with expired STS tokens; use an instance profile. |
| Container exits immediately | `docker logs <name>`; the startup checks fail loudly and name the problem. |
| Shard stuck at 0% | Normal for the first ~10 minutes: the bucket is listed once to reconcile link casing before any video is decoded. |

Nothing in this pipeline deletes S3 objects, and nothing writes outside
`flicker_results/` — both are enforced in code, in `src/data/publish.py`.
