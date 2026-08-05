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
FLICKER_KEY="visionlab/visionlab/outbound/India_Ahmedabad_Mirana_AssemblyLine_004_998.mp4"
BUCKET="humyn-data-partners-prod"

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; exit 1; }
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
run()  { uv run python main.py "${PASSTHRU[@]}" "$@" 2>&1 | grep -vE 'objc\[|libav.*dylib|may cause spurious'; }

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
resolver = S3Settings.from_config(load_config("configs/detector.yaml")).resolver()
resolver.verify_readable("s3://$BUCKET/$FLICKER_KEY")
print("    HEAD succeeded")
PY
pass "object is readable"

step 4 "Decode and score one known-flickering video over HTTP range reads"
echo "    (~1-2 min: signs a URL, then streams only the sampled windows)"
run "s3://$BUCKET/$FLICKER_KEY" --output "$WORK/one.csv" | tail -4
[ -s "$WORK/one.csv" ] || fail "no output written"
uv run python - <<PY || fail "scored row is malformed"
import csv
row = next(csv.DictReader(open("$WORK/one.csv")))
assert row["status"] == "ok", row.get("error")
print(f"    score={row['flicker_score'][:6]} band={row['severity_band']} "
      f"route={row['route']} worst={row['worst_segment_start']}-{row['worst_segment_end']}s")
print(f"    illuminant={row['illuminant_score'][:6]} band_det={row['rolling_band_score'][:6]} "
      f"awb={row['awb_score'][:6]}  ({row['processing_time_seconds'][:5]}s)")
assert row["severity_band"] != "none", (
    "reference flicker video scored 'none' -- detection or thresholds need review")
PY
pass "known-flickering video scored above the clean band"

step 5 "Parallel batch from the live prefix"
run --s3-prefix --output "$WORK/batch.csv" --limit 4 --workers 4 | grep -E "Progress|Finished|Rate:|videos/s" | tail -4
ok=$(uv run python -c "
import csv; rows=list(csv.DictReader(open('$WORK/batch.csv')))
print(sum(r['status']=='ok' for r in rows))")
[ "$ok" -gt 0 ] || fail "no video completed; see error column in $WORK/batch.csv"
pass "$ok/4 videos scored in parallel"

step 6 "Resume does no rework and keeps one row per key"
before=$(md5 -q "$WORK/batch.csv" 2>/dev/null || md5sum "$WORK/batch.csv" | cut -d' ' -f1)
run --s3-prefix --output "$WORK/batch.csv" --limit 4 --workers 4 --resume | grep -E "Resuming|No entries" | tail -2
after=$(md5 -q "$WORK/batch.csv" 2>/dev/null || md5sum "$WORK/batch.csv" | cut -d' ' -f1)
[ "$before" = "$after" ] || fail "resume rewrote already-scored rows"
pass "resume was a no-op on completed work"

printf '\n\033[32mAll checks passed.\033[0m Ready for a full run:\n'
printf '  uv run python main.py --s3-prefix --output output/flag_manifest.csv --workers 8 --resume\n\n'
