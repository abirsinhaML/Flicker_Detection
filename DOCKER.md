# Running the flicker detector in Docker

Hand-off notes for whoever operates the fleet. One container per instance, each
given a row range of the link sheet; the six ranges cover the corpus with no
overlap.

**Not yet built or run anywhere.** The Dockerfile is written against verified
facts about the dependencies, but no `docker build` has been executed — expect to
iterate once on the first build.

---

## What the job is

Score 140,519 videos (28,216 hours, read from S3) for flicker artifacts, and write
per-window results back to S3. Roughly 4,700 hours of video per instance across
six instances.

- **Reads** `s3://prod-egocentric-humyn-data/raw/…` — the corpus
- **Writes** `s3://stage-egocentric-humyn-data/flicker_results/` — the results
- Also writes results to a local volume, which is the source of truth; the S3 copy
  is derived
- Decode is ~94% of the cost, so this is GPU-bound on NVDEC when a GPU is present
  and CPU-bound otherwise. It runs correctly either way.

Expect roughly **20 days per instance** on a single-GPU box. It is safe to stop
and restart at any point: `--resume` skips completed videos.

---

## Host requirements

| | |
|---|---|
| Docker | any recent version |
| GPU (optional but ~6× faster) | NVIDIA driver + [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run with `--gpus all` |
| Disk | ~20 GB per instance for results, on the mounted `output/` volume |
| RAM | ~500 MB per worker; a 16-core box at the default 32 workers wants ~16 GB |
| Credentials | see below — an **IAM instance profile** is strongly preferred |

Without a GPU it still works: the pipeline probes NVDEC, logs one line, and falls
back to software decode.

---

## Credentials: use an instance profile

The container reads the corpus from one bucket and writes results to another, so
whichever identity it runs as needs both:

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

Attach that as an **instance profile** to each EC2 instance and pass no
credentials to the container at all — boto3 finds them via the metadata service
and refreshes them automatically. This matters because shards run for ~20 days
and exported STS/SSO tokens expire in hours; when they do, the batch stops being
able to read the corpus, not just to upload.

> **IMDSv2 hop limit — the one Docker-specific gotcha.** A container is one
> network hop further from the metadata service than the host, and the default
> hop limit of 1 silently blocks it. Raise it to 2 or the container will report
> `Unable to locate credentials` even with a role attached:
>
> ```bash
> aws ec2 modify-instance-metadata-options \
>   --instance-id i-xxxxxxxx \
>   --http-put-response-hop-limit 2 \
>   --http-tokens required
> ```

Static keys work too, if you must: `-e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY
-e AWS_SESSION_TOKEN`. Only viable with long-lived keys, for the expiry reason above.

---

## Build

From the repository root:

```bash
docker build -t flicker:0.2.0 .
```

For a fleet, build once and push to ECR so the instances pull rather than each
building:

```bash
ACCOUNT=<aws-account-id>; REGION=ap-south-1; REPO=flicker
aws ecr create-repository --repository-name $REPO --region $REGION
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker tag flicker:0.2.0 $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0
docker push $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0
```

On each instance:

```bash
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker pull $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0
```

---

## Run

Two volumes are required. The link sheet is **not** baked into the image, and
results must outlive the container:

```bash
mkdir -p /data/input /data/output
# copy input/all_s3links_updated.xlsx (4 MB, from the repo) to /data/input/

docker run -d \
  --name flicker \
  --restart unless-stopped \
  --gpus all \
  --init \
  -v /data/input:/app/input:ro \
  -v /data/output:/app/output \
  flicker:0.2.0 --from-row 0 --to-row 23170
```

Container arguments are the shard's row range. Everything else — output paths,
worker count, thread pinning, `--resume`, the S3 destination — is already handled
inside.

### The six shards

One line per instance. Ranges are inclusive and tile the sheet exactly.

| instance | arguments | rows | hours |
|---|---|---|---|
| 0 | `--from-row 0 --to-row 23170` | 23,171 | 4,703 |
| 1 | `--from-row 23171 --to-row 48960` | 25,790 | 4,702 |
| 2 | `--from-row 48961 --to-row 72553` | 23,593 | 4,702 |
| 3 | `--from-row 72554 --to-row 94575` | 22,022 | 4,703 |
| 4 | `--from-row 94576 --to-row 119113` | 24,538 | 4,703 |
| 5 | `--from-row 119114 --to-row 140518` | 21,405 | 4,703 |

Row counts differ on purpose: the split is balanced on *hours of video*, because
decode dominates the cost and hours predict runtime. Balanced by row count
instead, the heaviest shard would carry 1.20× the hours of the lightest and that
instance would still be running when the others finished.

Regenerate the table if the sheet changes:

```bash
docker run --rm -v /data/input:/app/input:ro --entrypoint python \
  flicker:0.2.0 main.py --links --print-shards 6
```

### Tuning

| env var | default | when to change |
|---|---|---|
| `WORKERS` | `2 × nproc` | With a GPU, NVDEC saturates near 3 concurrent streams; 32 workers mostly queue and each holds ~500 MB. Try 6 and compare the logged `Rate:`. Also set this explicitly if you limit the container with `--cpus`, since `nproc` still reports the host's count. |
| `BACKEND` | `auto` | `cpu` to force software decode |
| `CONFIG` | `configs/detector_1.yaml` | |

```bash
docker run -d ... -e WORKERS=6 flicker:0.2.0 --from-row 0 --to-row 23170
```

---

## Monitor

```bash
docker logs -f flicker                       # live output
docker exec flicker python scripts/check_progress.py
```

`check_progress.py` prints, per shard, how many videos are done out of the
resolvable total, the recent rate, and an ETA:

```
rows_0000000-0023170  rows 0..23170
  [##......................................]   5.2%
  scored     1,204 of 22,993 resolvable   (1,201 ok, 3 error)
  rate       98 videos/hour over the last 4.1h
  remaining  21,789 videos -> ~9.3d at this rate
```

Confirm on the first run that both of these appear in `docker logs`:

- `decode=cuda` on the metadata lines — otherwise it silently fell back to
  software and will take ~6× the CPU
- `Verified write access to s3://stage-egocentric-humyn-data/flicker_results/`

Check GPU saturation from the host with `nvidia-smi dmon`. `dec` should sit near
100%; if it is low while the CPU is pegged, see `decode.gpu_resize` in
`configs/detector_1.yaml`.

---

## Stop and restart

```bash
docker stop -t 60 flicker     # allow time to unwind the worker pool
docker start flicker          # resumes from the last completed video
```

`--resume` is always on. Resume granularity is **per video**, so videos that were
mid-decode when stopped are redone from the start — a few minutes of work, not
hours. Give `docker stop` a generous timeout: the default 10 s is usually not
enough to finish flushing, and although a hard kill loses nothing already written,
a clean exit is tidier.

Restarting the instance is fine too — `--restart unless-stopped` brings the
container back and it resumes.

---

## Output

Results appear in the mounted volume and, in parallel, in S3:

```
/data/output/
├── shards/rows_0000000-0023170.jsonl    every window of every video (the durable record)
├── shards/rows_0000000-0023170.csv      one row per video
├── shards/rows_0000000-0023170.log      the run log
└── window_metrics/raw/<key path>.csv    one CSV per video, at the video's own key

s3://stage-egocentric-humyn-data/flicker_results/
├── shards/rows_0000000-0023170.jsonl
├── shards/rows_0000000-0023170.csv
└── window_metrics/raw/<key path>.csv
```

**Per-video CSVs upload as each video completes. The shard `.jsonl` and `.csv`
upload only when the batch finishes** — they are appended to for the whole run, and
re-sending a multi-gigabyte file after every video would cost more than the batch.
On a 20-day shard that leaves the durable record un-copied for the duration, so
sync hourly from the host:

```cron
0 * * * * cd /data && /usr/bin/flock -n /tmp/flicker_upload.lock \
  docker exec flicker python upload_output.py --skip-existing --skip detector.log \
  >> /var/log/flicker_upload.log 2>&1
```

`--skip-existing` compares object size and skips unchanged files, so after the
first pass it only sends what actually grew.

The six instances write six shard files into the same S3 prefix and share
`window_metrics/`, which is safe: those files are keyed by video, so no two
instances can ever write the same one.

---

## Troubleshooting

| symptom | cause |
|---|---|
| `Unable to locate credentials` | No instance profile, or the IMDSv2 hop limit is 1. See above. |
| `cannot write to s3://…` at startup | Missing `s3:PutObject` on the results prefix, or a region mismatch — the error names which. Nothing is decoded before this check passes. |
| `decode=cpu` with a GPU present | `--gpus all` missing, or `nvidia-container-toolkit` not installed on the host. Runs correctly, ~6× more CPU. |
| `refusing to start: N main.py process(es) already running` | A previous run is still alive inside the container. `docker exec flicker scripts/stop_batch.sh`. |
| `S3 credentials are missing or expired` | The preflight refused to start. Expected with expired STS tokens; use an instance profile. |
| Container exits immediately | Read `docker logs flicker`; the startup checks fail loudly and name the problem. |

Nothing in this pipeline deletes S3 objects, and nothing writes outside
`flicker_results/` — both are enforced in code, in `src/data/publish.py`.
