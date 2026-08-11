#!/usr/bin/env bash
# Verify the S3 pipeline end to end, cheapest check first.
#
# Usage:
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...
#   ./scripts/verify_s3.sh
#   ./scripts/verify_s3.sh --aws-profile humyn      # or use a named profile
#
# Each check builds on the previous one, so the first failure tells you which
# layer is broken: credentials, list permission, read permission, or decode.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

PASSTHRU=("$@")
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# A video the assignment cites as visibly flickering: the strongest single
# end-to-end signal that detection still works when read over the network.
CONFIG="${CONFIG:-configs/detector_1.yaml}"
# A corpus video measured to score `extreme`: the strongest single end-to-end
# signal that detection still works when read over the network.
FLICKER_KEY="raw/Delhi_ZetWork/2026-06-17/HAA344/GX010091.MP4"
BUCKET="prod-egocentric-humyn-data"

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; exit 1; }
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
run()  { uv run python main.py --config "$CONFIG" "${PASSTHRU[@]}" "$@" 2>&1 | grep -vE 'objc\[|libav.*dylib|may cause spurious'; }

step 1 "Credentials resolve through the standard AWS chain"
uv run python - <<'PY' || fail "No usable credentials. Re-export them, or pass --aws-profile NAME."
import sys, boto3
from botocore.exceptions import BotoCoreError, ClientError
try:
    identity = boto3.Session().client("sts").get_caller_identity()
except (ClientError, BotoCoreError) as error:
    sys.exit(f"    {type(error).__name__}: {error}")
print(f"    account={identity['Account']}")
print(f"    arn={identity['Arn']}")
PY
pass "credentials accepted by STS"

step 2 "List permission and region (exercises verify_access + streamed listing)"
run --s3-prefix --write-manifest "$WORK/manifest.csv" --limit 25 | tail -3
rows=$(($(wc -l < "$WORK/manifest.csv") - 1))
[ "$rows" -gt 0 ] || fail "listing returned no videos: check s3.prefix and s3.region in configs/detector.yaml"
grep -q 'X-Amz' "$WORK/manifest.csv" && fail "manifest contains a signature; it must hold only s3:// URIs"
pass "listed $rows videos into a credential-free manifest"

step 3 "Read permission on one object (the batch preflight HEAD)"
uv run python - <<PY || fail "cannot read the object. Likely a region mismatch (SigV4 is region-scoped) or a missing s3:GetObject grant."
from src.data.s3_source import S3Settings
from main import load_config
resolver = S3Settings.from_config(load_config("$CONFIG")).resolver()
resolver.verify_readable("s3://$BUCKET/$FLICKER_KEY")
print("    HEAD succeeded")
PY
pass "object is readable"

step 4 "Decode and score one known-flickering video over HTTP range reads"
echo "    (signs a URL, then streams the whole video: windows are contiguous)"
run "s3://$BUCKET/$FLICKER_KEY" \
    --output "$WORK/one.jsonl" \
    --video-csv "$WORK/one.csv" \
    --window-dir "$WORK/window_metrics" | tail -5
[ -s "$WORK/one.jsonl" ] || fail "no output written"
uv run python - <<PY || fail "scored record is malformed"
import json
from src.core.records import window_csv_path

record = json.loads(open("$WORK/one.jsonl").read().strip())
assert record["status"] == "ok", record.get("error")
aggregate, windows = record["aggregate"], record["windows"]
print(f"    score={aggregate['flicker_score']:.4f} band={aggregate['severity_band']} "
      f"route={aggregate['route']} "
      f"worst={aggregate['worst_segment_start']:.1f}-{aggregate['worst_segment_end']:.1f}s")
print(f"    windows={aggregate['total_windows']} bands={aggregate['band_counts']}")

# Every reported column must reconcile with the windows it summarises.
scores = [w["score"] for w in windows]
assert len(windows) == aggregate["total_windows"], "window count disagrees with the rollup"
assert abs(aggregate["flicker_score"] - max(scores)) < 1e-9, (
    "flicker_score is not the maximum of its windows")
gaps = [b["start_time"] - a["end_time"] for a, b in zip(windows, windows[1:])]
assert not gaps or max(gaps) <= 1e-9, "schedule left a gap; windows should be contiguous"

# The per-video CSV must exist at the video's own key path.
per_video = window_csv_path(record["video_key"], "$WORK/window_metrics")
assert per_video.is_file(), f"no window CSV at {per_video}"
rows = sum(1 for _ in per_video.open()) - 1
assert rows == len(windows), f"window CSV has {rows} rows for {len(windows)} windows"
print(f"    window csv: {per_video.name} ({rows} rows)")

assert aggregate["severity_band"] != "none", (
    "reference flicker video scored 'none' -- detection or thresholds need review")
PY
pass "known-flickering video scored above the clean band, and its output reconciles"

step 5 "Parallel batch from the link sheet, over a row range"
run --links --from-row 0 --to-row 3 \
    --output "$WORK/batch.jsonl" --no-window-dir --no-video-csv --workers 4 \
    | grep -E "rows |Link resolution|Progress|Finished|Rate:" | tail -5
ok=$(uv run python -c "
import json
rows = [json.loads(line) for line in open('$WORK/batch.jsonl')]
print(sum(r['status'] == 'ok' for r in rows))")
[ "$ok" -gt 0 ] || fail "no video completed; see the error field in $WORK/batch.jsonl"
pass "$ok/4 videos scored in parallel from rows 0..3"

step 6 "Resume does no rework and keeps one record per key"
before=$(md5sum "$WORK/batch.jsonl" | cut -d' ' -f1)
run --links --from-row 0 --to-row 3 \
    --output "$WORK/batch.jsonl" --no-window-dir --no-video-csv --workers 4 --resume \
    | grep -E "Resuming|No entries" | tail -2
after=$(md5sum "$WORK/batch.jsonl" | cut -d' ' -f1)
[ "$before" = "$after" ] || fail "resume rewrote already-scored records"
pass "resume was a no-op on completed work"

printf '\n\033[32mAll checks passed.\033[0m Ready for a full run:\n'
printf '  scripts/run_batch.sh                              # whole sheet\n'
printf '  scripts/run_batch.sh --from-row 0 --to-row 14051  # one shard of ten\n\n'
printf 'Shard ranges for a fleet:\n'
printf '  uv run python main.py --links --print-shards 10\n\n'
