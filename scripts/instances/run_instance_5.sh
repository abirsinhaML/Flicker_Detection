#!/usr/bin/env bash
# Instance 5 of 6: rows 119114..140518 of input/all_s3links_updated.xlsx.
#
#   21,405 rows, 4,703 hours of video (16.7% of the corpus)
#
# One of six scripts, regenerate with:
#   .venv/bin/python main.py --links --print-shards 6
#
# The six ranges are balanced on DURATION, not row count, because decode is ~94%
# of the cost, so hours predict runtime and row counts do not.  Split by row count
# the heaviest of six shards carries 1.20x the hours of the lightest, and that
# instance is still running when the others have finished.  Balanced this way the
# six differ by 1 hour in 4,703.  The row counts therefore differ on purpose.
#
# Ranges are INCLUSIVE and tile the sheet exactly: this one ends at the row before
# instance 6 begins, and the six together cover all 140,519 rows with no
# overlap and no gap.  Instances share nothing at run time, so start them in any
# order and restart any one alone.
#
# Run it:
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...
#   scripts/instances/run_instance_5.sh
#
# Detached, surviving logout:
#   nohup scripts/instances/run_instance_5.sh > /dev/null 2>&1 &
#
# Stop it:  scripts/stop_batch.sh          (never `kill <pid>`)
# Watch it: tail -f output/shards/rows_0119114-0140518.log
#
# Results land under output/ on THIS machine:
#   output/shards/rows_0119114-0140518.jsonl   every window of every video scored here
#   output/shards/rows_0119114-0140518.csv     one row per video
#   output/shards/rows_0119114-0140518.log     this run's log
#   output/window_metrics/...     one CSV per video, at the video's own key path
#
# Collect the six machines' output/ directories afterwards and merge:
#   .venv/bin/python scripts/export_windows.py output/shards/*.jsonl \
#       --videos output/flag_manifest.csv
#
# --resume is already on, so re-running this exact command after a crash, an
# expired token, or a reboot continues from the last completed video.
#
# Results are also copied to s3.output in configs/detector_1.yaml -- currently
#   s3://stage-egocentric-humyn-data/flicker_results/
# mirroring the local output/ layout.  Write access is checked once before any
# decoding, so a missing grant stops the run immediately rather than after hours.
# Pass --no-s3-output to keep this shard local only, or --s3-output-dry-run to
# print the destination keys without uploading.

set -euo pipefail
cd "$(dirname "$0")/../.."

# Cleared rather than inherited: an exported WORKERS from an earlier command
# silently overrides run_batch.sh's default and has already cost one run its
# parallelism.  run_batch.sh uses 2x the core count.
unset WORKERS

exec scripts/run_batch.sh \
    --from-row 119114 \
    --to-row 140518 \
    "$@"
