#!/usr/bin/env bash
# Stop the batch without leaving its workers behind.
#
# `kill <parent_pid>` sends SIGTERM to one process.  Python's default handler exits
# immediately -- no finally, no atexit, no context-manager unwind -- so the
# `with ProcessPoolExecutor(...)` block in run_batch never shuts the pool down.
# Every worker is reparented to init and keeps decoding videos whose results have
# nowhere to go.  Four runs on this box ended that way, once leaving 32 orphans
# competing for cores with the run that replaced them.
#
# Signalling the process *group* (the leading dash) reaches the parent and every
# worker at once, which is what Ctrl-C in the pane does.

set -uo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/batch_procs.sh
. "$(dirname "$0")/batch_procs.sh"

if [ "$(batch_count)" -eq 0 ]; then
    echo "nothing running"
    exit 0
fi

PG=$(batch_pgid)
echo "stopping process group $PG ($(batch_count) processes)"

# SIGINT first: the parent unwinds its pool and workers raise KeyboardInterrupt.
# Rows already flushed to the CSV survive, so --resume picks up from there.
kill -INT -"$PG" 2>/dev/null
for _ in $(seq 15); do
    [ "$(batch_count)" -eq 0 ] && break
    sleep 1
done

# A worker blocked inside NVDEC or an S3 read does not always take SIGINT.
if [ "$(batch_count)" -gt 0 ]; then
    echo "  $(batch_count) still up after SIGINT; escalating"
    kill -TERM -"$PG" 2>/dev/null
    sleep 5
    [ "$(batch_count)" -gt 0 ] && { kill -KILL -"$PG" 2>/dev/null; sleep 2; }
fi

# Orphans from *earlier* runs are in a different group, so sweep them separately.
strays=$(batch_orphans)
if [ -n "$strays" ]; then
    echo "  sweeping $(echo "$strays" | wc -w) orphan(s) from earlier runs"
    # shellcheck disable=SC2086
    kill -KILL $strays 2>/dev/null
    sleep 2
fi

if [ "$(batch_count)" -eq 0 ]; then
    echo "clean"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
        | sed 's/^/  gpu: /'
else
    echo "STILL RUNNING:" >&2
    batch_show >&2
    exit 1
fi
