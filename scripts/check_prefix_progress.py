#!/usr/bin/env python3
"""Check processing progress across multiple S3 prefixes.

Usage:
    python scripts/check_prefix_progress.py output/flag_manifest.csv configs/detector_1.yaml
"""

import argparse
import contextlib
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# How many of the most recent completions define the "current" rate. Averaging
# over the whole run hides a slowdown, and hides the speedup after a config
# change too, because early hours keep their weight forever.
RECENT_WINDOW = 200


def print_throughput(completions: list[datetime], processing_times: list[float]) -> None:
    """Report how fast rows are being produced, from their completion stamps.

    Rows written before the ``completed_at`` column existed carry no stamp and
    are skipped, so an older manifest reports on whatever part of itself is
    timestamped rather than reporting nothing.
    """
    if len(completions) < 2:
        print("\nThroughput: needs at least two timestamped rows "
              "(older rows predate the completed_at column).")
        return

    completions.sort()
    span_hours = (completions[-1] - completions[0]).total_seconds() / 3600.0
    recent = completions[-RECENT_WINDOW:]
    recent_hours = (recent[-1] - recent[0]).total_seconds() / 3600.0

    print(f"\n{'Throughput':^80}")
    print(f"{'-' * 80}")
    print(f"  Timestamped rows:   {len(completions):,}")
    print(f"  First completion:   {completions[0].isoformat()}")
    print(f"  Last completion:    {completions[-1].isoformat()}")
    print(f"  Elapsed:            {span_hours:.2f} h")
    if span_hours > 0:
        overall = len(completions) / span_hours
        print(f"  Overall rate:       {overall:,.1f} videos/hour")
    if len(recent) > 1 and recent_hours > 0:
        print(f"  Recent rate:        {len(recent) / recent_hours:,.1f} videos/hour "
              f"(last {len(recent)})")
    if processing_times:
        mean_seconds = sum(processing_times) / len(processing_times)
        print(f"  Mean per video:     {mean_seconds:.1f} s of worker time")
        # Wall-clock rate over worker-time per video is the effective parallelism
        # actually achieved, which is the number worth comparing to --workers.
        if span_hours > 0:
            achieved = (len(completions) / span_hours) * mean_seconds / 3600.0
            print(f"  Effective workers:  {achieved:.1f} (vs the --workers you set)")


def extract_prefix_from_key(key: str, prefixes: list[str]) -> str:
    """Find which configured prefix this key belongs to."""
    for prefix in prefixes:
        if key.startswith(prefix):
            return prefix
    # If no match, use the directory up to the filename
    parts = key.split('/')
    if len(parts) > 1:
        return '/'.join(parts[:-1]) + '/'
    return 'unknown'


def main():
    parser = argparse.ArgumentParser(description='Check flicker detection progress by prefix')
    parser.add_argument('output_csv', help='Path to the output flag manifest CSV')
    parser.add_argument('config', help='Path to the detector config YAML')
    args = parser.parse_args()

    # Load configured prefixes
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    configured_prefixes = config.get('s3', {}).get('prefixes', [])
    if not configured_prefixes:
        single_prefix = config.get('s3', {}).get('prefix', '')
        configured_prefixes = [single_prefix] if single_prefix else []
    
    if not configured_prefixes:
        print("No prefixes configured in the YAML file.")
        return

    # Count by prefix and status
    stats = defaultdict(lambda: {'ok': 0, 'error': 0, 'total': 0})
    completions: list[datetime] = []
    processing_times: list[float] = []

    output_path = Path(args.output_csv)
    if not output_path.exists():
        print(f"Output file not found: {args.output_csv}")
        print("No processing has started yet.")
        return
    
    with open(output_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get('video_key', '')
            status = row.get('status', '')
            
            if not key:
                continue
                
            prefix = extract_prefix_from_key(key, configured_prefixes)
            stats[prefix]['total'] += 1
            if status == 'ok':
                stats[prefix]['ok'] += 1
            elif status == 'error':
                stats[prefix]['error'] += 1

            # A malformed stamp costs that row from the rate, not the report.
            stamp = (row.get('completed_at') or '').strip()
            if stamp:
                with contextlib.suppress(ValueError):
                    completions.append(datetime.fromisoformat(stamp))
            elapsed = (row.get('processing_time_seconds') or '').strip()
            if elapsed:
                with contextlib.suppress(ValueError):
                    processing_times.append(float(elapsed))

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"{'Prefix Progress Summary':^80}")
    print(f"{'=' * 80}")
    print(f"{'Prefix':<50} {'Total':>8} {'OK':>8} {'Error':>8} {'%'}")
    print(f"{'-' * 80}")
    
    total_videos = 0
    total_ok = 0
    total_error = 0
    
    for prefix in configured_prefixes:
        s = stats[prefix]
        total_videos += s['total']
        total_ok += s['ok']
        total_error += s['error']
        
        pct = (s['ok'] / s['total'] * 100) if s['total'] > 0 else 0
        status_mark = '✓' if s['total'] > 0 and s['error'] == 0 and s['ok'] == s['total'] else ''
        
        print(f"{prefix:<50} {s['total']:>8} {s['ok']:>8} {s['error']:>8} {pct:>5.1f}% {status_mark}")
    
    # Check for unrecognized prefixes
    unrecognized = set(stats.keys()) - set(configured_prefixes)
    if unrecognized:
        print(f"\n{'Unrecognized prefixes (not in config):':^80}")
        for prefix in sorted(unrecognized):
            s = stats[prefix]
            pct = (s['ok'] / s['total'] * 100) if s['total'] > 0 else 0
            print(f"{prefix:<50} {s['total']:>8} {s['ok']:>8} {s['error']:>8} {pct:>5.1f}%")
            total_videos += s['total']
            total_ok += s['ok']
            total_error += s['error']
    
    print(f"{'-' * 80}")
    total_pct = (total_ok / total_videos * 100) if total_videos > 0 else 0
    print(f"{'TOTAL':<50} {total_videos:>8} {total_ok:>8} {total_error:>8} {total_pct:>5.1f}%")
    print(f"{'=' * 80}\n")
    
    # Summary statistics
    if total_videos > 0:
        print(f"Summary:")
        print(f"  - Configured prefixes: {len(configured_prefixes)}")
        print(f"  - Prefixes with results: {len([p for p in configured_prefixes if stats[p]['total'] > 0])}")
        print(f"  - Total videos processed: {total_videos:,}")
        print(f"  - Success rate: {total_pct:.1f}%")
        if total_error > 0:
            print(f"  - Failed videos: {total_error} (will be retried with --resume)")

    print_throughput(completions, processing_times)


if __name__ == '__main__':
    main()
