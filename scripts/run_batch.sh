#!/usr/bin/env bash
# Launch the scoring batch with the thread and worker settings that were measured
# on this box, and refuse to start in the states that have silently wasted runs.
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
#   WORKERS=32                The box has 16 cores, but the right count is not 16.
#                             Fully pinned at 16 workers it measured 83% CPU with
#                             each worker at 82% of a core: workers block on S3
#                             about 18% of the time, and 16 of them cannot fill
#                             those gaps, leaving ~2.7 cores idle.  Oversubscribing
#                             covers the stall.  32 is the untested end of that --
#                             the older "91% CPU, no worker above 48%" reading was
#                             taken with OpenCV *unpinned*, when ~500 threads were
#                             thrashing, so it does not describe this config.
#                             Check CPU and windows/s at 32 and adjust.
#
# Stop it with Ctrl-C in this pane, or scripts/stop_batch.sh.  Never `kill <pid>`:
# Python takes SIGTERM without unwinding, so the pool is never shut down and every
# worker is reparented to init and keeps decoding.  That has happened four times.

set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/batch_procs.sh
. "$(dirname "$0")/batch_procs.sh"

CONFIG="${CONFIG:-configs/detector_1.yaml}"
OUTPUT="${OUTPUT:-output/flag_manifest.csv}"
WORKERS="${WORKERS:-32}"

# A second batch writing the same CSV, or a previous run's orphans stealing cores,
# both look like "the GPU is slow" from the outside.  Fail loudly instead.
existing=$(batch_count)
if [ "$existing" -gt 0 ]; then
    echo "refusing to start: $existing main.py process(es) already running" >&2
    batch_show >&2
    echo >&2
    echo "stop them first:  scripts/stop_batch.sh" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

# Signing is offline and succeeds even with expired credentials, so without this
# check the failure would surface as one error row per video instead of once here.
echo "checking S3 credentials..."
if ! .venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from main import load_config
from src.data.s3_source import S3Settings
S3Settings.from_config(load_config('$CONFIG')).catalog().verify_access()
" 2>/dev/null; then
    echo "refusing to start: S3 credentials are missing or expired" >&2
    echo "  export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=..." >&2
    exit 1
fi

echo "starting: workers=$WORKERS config=$CONFIG output=$OUTPUT"
echo "  threads pinned: OMP/OPENBLAS/MKL/NUMEXPR/OPENCV = 1"
echo "  stop with Ctrl-C in this pane (not kill <pid>)"
echo

exec .venv/bin/python3 main.py \
    --s3-prefix \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --workers "$WORKERS" \
    --resume
