#!/usr/bin/env bash
# Launch a scoring batch with the thread and worker settings measured on this box,
# and refuse to start in the states that have silently wasted runs.
#
# Usage:
#   scripts/run_batch.sh                                   # whole link sheet
#   scripts/run_batch.sh --from-row 0 --to-row 99          # one shard
#   WORKERS=4 scripts/run_batch.sh --from-row 100 --to-row 199
#   scripts/run_batch.sh --manifest data/manifest.csv --from-row 0 --to-row 999
#
# Print the shard ranges for a fleet instead of typing them:
#   .venv/bin/python main.py --links --config configs/detector_1.yaml --print-shards 10
#
# Extra arguments pass straight through to main.py, so --limit, --decode-backend
# and --s3-output all work here.
#
# OUTPUT NAMING, AND WHY IT MATTERS FOR A FLEET.  Each run writes its own JSONL
# and its own rollup CSV, named after the row range, because both are opened with
# "w" and two instances sharing one would truncate each other's results.  The
# per-video window CSVs are shared on purpose: one file per video key, so
# instances working different rows can never collide and the tree assembles
# itself.
#
# Why each export is here, measured rather than assumed:
#
#   OMP/OPENBLAS/MKL/NUMEXPR  NumPy is on scipy-openblas, which spawns one thread
#                             per core in every worker.  Unpinned, the detector
#                             stage took 20.4 ms wall / 308 ms CPU across 15.1
#                             threads; pinned it takes 10.6 ms wall / 10.6 ms CPU.
#                             Faster *and* 29x cheaper, because the thread thrash
#                             cost more than the parallelism bought.  Scores are
#                             bit-identical either way (0.09233131259679794).
#
#   OPENCV_FOR_THREADS_NUM    OpenCV keeps a pool the vars above do not touch;
#                             cv2.getNumThreads() stays at 16 without this.
#
#   WORKERS                   Two per core on both backends, so 32 on this
#                             16-core box.  Software decode wants the
#                             oversubscription outright: workers block on S3 about
#                             18% of the time and 16 of them leave ~2.7 cores idle.
#
#                             On the GPU path it is deliberately more than the
#                             decode engine can use.  Measured on this A10G, the
#                             engine is the constraint -- 100% busy with the SMs at
#                             8%, and five workers bought 2.5x a single worker
#                             rather than 5x -- so workers past a handful queue on
#                             it rather than adding throughput.  They are not free
#                             either: each holds a CUDA context of roughly 375 MB,
#                             so 32 is ~12 GB of the card's 23 GB.  Harmless here,
#                             worth knowing on a smaller card, and worth lowering
#                             if NVDEC starts refusing sessions.
#
#                             Override with WORKERS=N.  NOTE that an exported
#                             WORKERS from an earlier command wins over this
#                             default -- the resolved value and where it came from
#                             are printed and logged below, because a stale export
#                             is otherwise invisible.
#
# Stop it with Ctrl-C in this pane, or scripts/stop_batch.sh.  Never `kill <pid>`:
# Python takes SIGTERM without unwinding, so the pool is never shut down and every
# worker is reparented to init and keeps decoding.  That has happened four times.

set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/batch_procs.sh
. "$(dirname "$0")/batch_procs.sh"

CONFIG="${CONFIG:-configs/detector_1.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output}"
BACKEND="${BACKEND:-auto}"

# Pull the row range out of the passthrough so the output files can be named
# after it, while still forwarding it to main.py.
FROM_ROW=""
TO_ROW=""
ARGS=("$@")
for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --from-row)   FROM_ROW="${ARGS[i + 1]:-}" ;;
        --to-row)     TO_ROW="${ARGS[i + 1]:-}" ;;
        --from-row=*) FROM_ROW="${ARGS[i]#*=}" ;;
        --to-row=*)   TO_ROW="${ARGS[i]#*=}" ;;
    esac
done

if [ -n "$FROM_ROW" ] || [ -n "$TO_ROW" ]; then
    SHARD="rows_$(printf '%07d' "${FROM_ROW:-0}")-$(printf '%07d' "${TO_ROW:-9999999}")"
else
    SHARD="all"
fi
SHARD_DIR="$OUTPUT_ROOT/shards"
RECORDS="${RECORDS:-$SHARD_DIR/$SHARD.jsonl}"
VIDEO_CSV="${VIDEO_CSV:-$SHARD_DIR/$SHARD.csv}"
WINDOW_DIR="${WINDOW_DIR:-$OUTPUT_ROOT/window_metrics}"
LOG="${LOG:-$SHARD_DIR/$SHARD.log}"

# Two per core on both backends; see the note above.
#
# nproc is asked with OMP_NUM_THREADS and OMP_THREAD_LIMIT cleared, because
# coreutils nproc *honours* them: it returns the minimum of the real CPU count
# and OMP_NUM_THREADS.  This script exports OMP_NUM_THREADS=1 a few lines below,
# so any shell that has run it before -- or sourced it, or set the variable for
# any other reason -- reports one core, and the default silently collapses from
# 32 workers to 2.  That is exactly what happened on a 16-core box.  Clearing the
# variables for this one call keeps the answer affinity-aware (a genuinely
# restricted cpuset still counts correctly) while ignoring the thread pinning.
CORES=$(env -u OMP_NUM_THREADS -u OMP_THREAD_LIMIT nproc)

# WORKERS_ORIGIN is carried so the header can say where the number came from: an
# exported WORKERS from an earlier command silently beats this default, which has
# already cost one run its parallelism.
if [ -n "${WORKERS:-}" ]; then
    WORKERS_ORIGIN="WORKERS from the environment"
else
    WORKERS=$(( CORES * 2 ))
    WORKERS_ORIGIN="default: $CORES cores x 2"
fi

# A second batch writing the same shard, or a previous run's orphans stealing
# cores, both look like "the GPU is slow" from the outside.  Fail loudly instead.
existing=$(batch_count)
if [ "$existing" -gt 0 ] && [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    echo "refusing to start: $existing main.py process(es) already running" >&2
    batch_show >&2
    echo >&2
    echo "stop them first:  scripts/stop_batch.sh" >&2
    echo "to run several shards on ONE box deliberately, give each a distinct" >&2
    echo "--from-row/--to-row and set ALLOW_CONCURRENT=1" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

# Signing is offline and succeeds even with expired credentials, so without this
# check the failure would surface as one error row per video instead of once here.
#
# SKIP_S3_CHECK=1 bypasses it for the case the check gets wrong: a --manifest of
# local paths needs no S3 access at all, and refusing to start then blocks a run
# that would have worked.
if [ "${SKIP_S3_CHECK:-0}" = "1" ]; then
    echo "skipping the S3 credential check (SKIP_S3_CHECK=1)"
else
    echo "checking S3 credentials..."
    if ! .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from main import load_config
from src.data.s3_source import S3Settings
S3Settings.from_config(load_config('$CONFIG')).catalog().verify_access()
" 2>/dev/null; then
        echo "refusing to start: S3 credentials are missing or expired" >&2
        echo "  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=..." >&2
        echo "  (SKIP_S3_CHECK=1 bypasses this, for a --manifest of local paths)" >&2
        exit 1
    fi
fi

mkdir -p "$SHARD_DIR" "$WINDOW_DIR"

# --links with no value reads s3.links from the config.  An explicit --links,
# --manifest or --s3-prefix in the passthrough replaces it, since main.py takes
# exactly one input and would refuse two.
INPUT=(--links)
for arg in "$@"; do
    case "$arg" in
        --links|--links=*|--manifest|--manifest=*|--s3-prefix|--s3-prefix=*)
            INPUT=()
            ;;
    esac
done

# The header goes through tee as well.  It used to be echoed straight to the
# terminal while only python's output was logged, so the one number an operator
# most needs afterwards -- how many workers actually started -- was absent from
# the log, and a stale WORKERS export could not be diagnosed from it.
{
    echo "starting: shard=$SHARD backend=$BACKEND config=$CONFIG"
    echo "  workers:    $WORKERS   ($WORKERS_ORIGIN)"
    echo "  records:    $RECORDS"
    echo "  video csv:  $VIDEO_CSV"
    echo "  window dir: $WINDOW_DIR   (shared across shards; one file per video)"
    echo "  log:        $LOG"
    echo "  threads pinned: OMP/OPENBLAS/MKL/NUMEXPR/OPENCV = 1"
    echo "  stop with Ctrl-C in this pane (not kill <pid>)"
    echo
} | tee -a "$LOG"

# tee rather than exec-and-redirect, so a detached run still shows progress in
# the pane it was started from and keeps the same log across --resume.
.venv/bin/python3 main.py \
    "${INPUT[@]}" \
    --config "$CONFIG" \
    --output "$RECORDS" \
    --video-csv "$VIDEO_CSV" \
    --window-dir "$WINDOW_DIR" \
    --output-root "$OUTPUT_ROOT" \
    --decode-backend "$BACKEND" \
    --workers "$WORKERS" \
    --resume \
    "$@" 2>&1 | tee -a "$LOG"
