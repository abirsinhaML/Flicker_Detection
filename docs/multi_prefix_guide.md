# Multi-Prefix Processing Guide

## Overview

The flicker detector now supports processing multiple S3 prefixes in a single run. This is useful when you have a dataset distributed across multiple locations in the same S3 bucket.

## Configuration

In your detector YAML config file (`configs/detector_1.yaml`), you can specify multiple prefixes using the `prefixes` field:

```yaml
s3:
  bucket: prod-egocentric-humyn-data
  prefixes:
    - raw/2026-05-08/project-test/              # done
    - raw/2026-05-16/mirana_test/
    - raw/2026-05-16/project-test/
    - raw/collector_app/
    # ... more prefixes
  region: ap-south-1
```

### Fallback Behavior

If `prefixes` is not set or is empty, the system falls back to using the single `prefix` field:

```yaml
s3:
  bucket: prod-egocentric-humyn-data
  prefix: raw/2026-05-16/mirana_test/    # Used only if prefixes is empty
  region: ap-south-1
```

## Usage

### Process All Configured Prefixes

```bash
uv run python main.py --s3-prefix \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

When `--s3-prefix` is used without an argument, it processes all prefixes defined in the config file.

### Override with a Specific Prefix

You can still override the config by providing a specific prefix:

```bash
uv run python main.py --s3-prefix s3://prod-egocentric-humyn-data/raw/specific/prefix/ \
  --output output/flag_manifest.csv \
  --workers 8
```

### Create a Snapshot Manifest

To create a reproducible manifest from all prefixes:

```bash
uv run python main.py --s3-prefix --write-manifest data/full_dataset_manifest.csv
```

This will:
1. List all videos from all configured prefixes
2. Write them to a single manifest file
3. Exit without processing

You can then process the snapshot:

```bash
uv run python main.py --manifest data/full_dataset_manifest.csv \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume
```

## Resume Support

The `--resume` flag works seamlessly with multi-prefix processing:

- Already-processed videos are skipped regardless of which prefix they came from
- Failures are retried (useful for transient errors like expired credentials)
- The output file maintains exactly one row per `video_key`

This means you can safely interrupt a multi-prefix run and restart it:

```bash
# Start processing all prefixes
uv run python main.py --s3-prefix \
  --output output/flag_manifest.csv \
  --workers 8 \
  --resume

# If interrupted or credentials expire, just run the same command again
# It will pick up where it left off
```

## Processing Order

Prefixes are processed in the order they appear in the config file. Videos from all prefixes are streamed together, so:

- Memory usage stays constant regardless of total corpus size
- Workers process videos from any prefix as they become available
- The output is not grouped by prefix (videos are processed in parallel)

## Tracking Progress

To track which prefixes have been completed, you can:

1. Add comments in the config (e.g., `# done`)
2. Use the output CSV to see which videos have been processed
3. Monitor logs for prefix-specific messages

Example: checking how many videos from each prefix have been processed:

```bash
# Count processed videos per prefix pattern
awk -F, 'NR>1 && $2=="ok" {print $4}' output/flag_manifest.csv | \
  sed 's|/[^/]*$||' | \
  sort | uniq -c | sort -rn
```

## Notes

- All prefixes must be in the same bucket (specified by `s3.bucket`)
- The same AWS credentials are used for all prefixes
- The `s3.region` must match the bucket's region for all operations
