# Quick Reference: Multi-Prefix Processing

## Start/Resume Processing All Prefixes

```bash
uv run python main.py --s3-prefix \
  --config configs/detector_1.yaml \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

**Use this command every time:**
- First run: processes everything from scratch
- Subsequent runs: skips already-processed videos, retries failures
- After credentials expire: refresh credentials and run again
- After interruption: just run again

## Check Progress

```bash
uv run python scripts/check_prefix_progress.py \
  output/flag_manifest.csv \
  configs/detector_1.yaml
```

Shows which prefixes are done and overall completion percentage.

## Update Config After Completing a Prefix

Edit `configs/detector_1.yaml` and either:

**Option 1: Mark as done (keeps in list)**
```yaml
prefixes:
  - raw/2026-05-08/project-test/              # ✓ done
  - raw/2026-05-16/mirana_test/               # ✓ done
  - raw/2026-05-16/project-test/
```

**Option 2: Comment out (removes from processing)**
```yaml
prefixes:
  # - raw/2026-05-08/project-test/              # ✓ done
  # - raw/2026-05-16/mirana_test/               # ✓ done
  - raw/2026-05-16/project-test/
```

## Common Scenarios

### Credentials Expired Mid-Run

```bash
# 1. Refresh your AWS credentials
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_SESSION_TOKEN="..."

# 2. Resume (same command as always)
uv run python main.py --s3-prefix \
  --config configs/detector_1.yaml \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

### Process Was Killed

```bash
# Just restart with --resume
uv run python main.py --s3-prefix \
  --config configs/detector_1.yaml \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

### Want to Process Just One Prefix

```bash
# Override with specific prefix
uv run python main.py --s3-prefix raw/specific/prefix/ \
  --config configs/detector_1.yaml \
  --output output/specific_output.csv \
  --workers 8
```

Or temporarily comment out other prefixes in the config.

### Create Reproducible Snapshot First

```bash
# 1. Create manifest (doesn't process, just lists)
uv run python main.py --s3-prefix \
  --config configs/detector_1.yaml \
  --write-manifest data/full_dataset_manifest.csv

# 2. Process the fixed manifest
uv run python main.py \
  --manifest data/full_dataset_manifest.csv \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

## Tips

- **Always use `--resume`**: It's safe and saves time
- **Check progress regularly**: Run the progress script to see which prefixes are complete
- **Mark completed prefixes**: Update the config to track progress
- **One output file**: All prefixes write to the same CSV (one row per video)
- **Safe to interrupt**: Just restart with the same command

## Output Location

All results are written to: `output/flag_manifest.csv`

One row per video across all prefixes.

## Progress Tracking

The progress checker shows:
- Videos processed per prefix
- Success/error counts
- Completion percentage
- Which prefixes are fully done (✓)

## File Locations

- **Config**: `configs/detector_1.yaml` (edit here to mark progress)
- **Output**: `output/flag_manifest.csv` (all results)
- **Progress script**: `scripts/check_prefix_progress.py`
- **Documentation**: `docs/multi_prefix_guide.md`
