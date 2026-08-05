# Multi-Prefix Support Implementation Summary

## What Was Done

The flicker detection system has been enhanced to support processing multiple S3 prefixes in a single run, allowing you to process your entire distributed dataset without manually switching configurations.

## Changes Made

### 1. Code Changes (Already Completed by Previous Session)

#### `src/data/s3_source.py`
- Added `prefixes: tuple[str, ...]` field to `S3Settings`
- Added `catalogs()` method that returns a list of `S3VideoCatalog` objects
  - If `listing_root` is provided, uses that (single catalog)
  - If `prefixes` is configured, creates one catalog per prefix
  - Falls back to single `prefix` field if neither is set

#### `main.py`
- Updated `run_s3_prefix()` to use `catalogs()` and process all prefixes
  - Verifies access to all prefixes before starting
  - Chains entries from all catalogs into a single stream
  - Logs each prefix being processed
- Updated `snapshot_s3_manifest()` similarly to handle multiple prefixes
  - Creates a single manifest from all configured prefixes
  - Logs the count of prefixes processed

### 2. Configuration Update

#### `configs/detector_1.yaml`
- Added `prefixes` list with all 22 dataset locations
- Kept single `prefix` field as fallback
- Added helpful comments explaining usage
- Marked the first prefix as done: `# ✓ done`

```yaml
s3:
  bucket: prod-egocentric-humyn-data
  prefixes:
    - raw/2026-05-08/project-test/              # ✓ done
    - raw/2026-05-16/mirana_test/
    - raw/2026-05-16/project-test/
    # ... 19 more prefixes
  prefix: raw/2026-05-16/mirana_test/  # fallback
  region: ap-south-1
```

### 3. Documentation

#### `docs/multi_prefix_guide.md`
Complete guide covering:
- How to configure multiple prefixes
- Usage examples for processing and manifest creation
- Resume support explanation
- Progress tracking tips
- Processing order notes

#### `scripts/check_prefix_progress.py`
Utility script to check progress across all prefixes:
- Reads the output CSV and groups by prefix
- Shows count of OK/error videos per prefix
- Calculates completion percentage
- Identifies which prefixes are done
- Reports overall statistics

## How to Use

### Process All Prefixes

```bash
uv run python main.py --s3-prefix \
  --config configs/detector_1.yaml \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

### Check Progress

```bash
uv run python scripts/check_prefix_progress.py \
  output/flag_manifest.csv \
  configs/detector_1.yaml
```

Example output:
```
================================================================================
                         Prefix Progress Summary
================================================================================
Prefix                                               Total       OK    Error    %
--------------------------------------------------------------------------------
raw/2026-05-08/project-test/                           150      150        0 100.0% ✓
raw/2026-05-16/mirana_test/                             45       45        0 100.0% ✓
raw/2026-05-16/project-test/                             0        0        0   0.0%
raw/collector_app/                                       0        0        0   0.0%
...
```

### Mark Completed Prefixes

As prefixes complete, you can:
1. Add `# ✓ done` comment in the config (visual tracking)
2. Comment out completed prefixes to skip on next run:
   ```yaml
   # - raw/2026-05-08/project-test/              # ✓ done
   ```

### Create a Reproducible Snapshot

```bash
# Snapshot all prefixes to a single manifest
uv run python main.py --s3-prefix \
  --config configs/detector_1.yaml \
  --write-manifest data/full_dataset_manifest.csv

# Process the snapshot
uv run python main.py \
  --manifest data/full_dataset_manifest.csv \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

## Key Features

### Automatic Prefix Switching
- No manual intervention needed
- System processes all prefixes in sequence
- All results go to a single output file

### Resume Support
- Works across all prefixes
- Already-processed videos are skipped regardless of prefix
- Safe to interrupt and restart anytime
- Useful when credentials expire mid-batch

### Memory Efficiency
- Entries are streamed lazily from all prefixes
- Only a bounded number of videos are in flight at once
- Memory usage doesn't scale with corpus size

### Backward Compatible
- If `prefixes` is not set, falls back to single `prefix`
- Existing configs continue to work
- Can override with `--s3-prefix s3://bucket/specific/prefix/`

## Testing

Configuration parsing verified:
```bash
✓ Bucket: prod-egocentric-humyn-data
✓ Number of prefixes: 22
✓ All 22 prefixes loaded correctly
```

## Next Steps

1. **Run the full dataset**:
   ```bash
   uv run python main.py --s3-prefix \
     --config configs/detector_1.yaml \
     --output output/flag_manifest.csv \
     --workers 8 \
     --resume
   ```

2. **Monitor progress** periodically:
   ```bash
   uv run python scripts/check_prefix_progress.py \
     output/flag_manifest.csv \
     configs/detector_1.yaml
   ```

3. **Handle interruptions**:
   - If credentials expire: refresh them and rerun with `--resume`
   - If process is killed: just rerun with `--resume`
   - The system picks up where it left off

4. **Update the config** as prefixes complete:
   - Mark completed: `# ✓ done`
   - Or comment out to skip: `# - prefix/  # done`

## Benefits

- **No manual tracking needed**: One command processes everything
- **Crash resilient**: Resume support handles interruptions gracefully
- **Progress visibility**: Check which prefixes are done at any time
- **Single output file**: All results in one place for easy analysis
- **Efficient**: Constant memory usage regardless of dataset size
