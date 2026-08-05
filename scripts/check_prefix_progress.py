#!/usr/bin/env python3
"""Check processing progress across multiple S3 prefixes.

Usage:
    python scripts/check_prefix_progress.py output/flag_manifest.csv configs/detector_1.yaml
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import yaml


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


if __name__ == '__main__':
    main()
