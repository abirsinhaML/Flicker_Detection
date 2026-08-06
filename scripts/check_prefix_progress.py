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

# A pool of N workers starts N videos together and finishes them together, so
# rows land in clumps with quiet stretches between. Measuring inside one clump
# reports several times the true rate. Only a gap longer than this is taken to
# mean the batch itself stopped; anything shorter is that clumping and must be
# averaged over, not split on.
RUN_GAP_SECONDS = 300


def print_throughput(finished: list[tuple[datetime, float | None]]) -> None:
    """Report how fast rows are being produced, from their completion stamps.

    Rows written before the ``completed_at`` column existed carry no stamp and
    are skipped, so an older manifest reports on whatever part of itself is
    timestamped rather than reporting nothing.

    A manifest accumulates every run that ever wrote to it, and those runs are
    separated by however long the machine sat idle.  Averaging across that
    reports a rate no run achieved, so the figures below describe the most
    recent contiguous session only.
    """
    if len(finished) < 2:
        print("\nThroughput: needs at least two timestamped rows "
              "(older rows predate the completed_at column).")
        return

    finished.sort(key=lambda entry: entry[0])
    session = [finished[0]]
    for previous, current in zip(finished, finished[1:]):
        if (current[0] - previous[0]).total_seconds() > RUN_GAP_SECONDS:
            session = [current]
        else:
            session.append(current)

    span_hours = (session[-1][0] - session[0][0]).total_seconds() / 3600.0
    earlier = len(finished) - len(session)
    # The service time has to come from the same rows as the rate. Drawing it
    # from the whole manifest mixes in runs at other worker counts -- and every
    # row a slower configuration ever wrote -- which is how this came out at
    # three times the workers that existed.
    session_times = [seconds for _, seconds in session if seconds is not None]

    print(f"\n{'Throughput (successful videos, latest session)':^80}")
    print(f"{'-' * 80}")
    print(f"  Completed:          {len(session):,}"
          f"{f'  (+{earlier:,} from earlier sessions)' if earlier else ''}")
    print(f"  Session started:    {session[0][0].isoformat()}")
    print(f"  Last completion:    {session[-1][0].isoformat()}")
    print(f"  Elapsed:            {span_hours * 60:.1f} min")
    if span_hours <= 0:
        print("  Rate:               too early to measure")
        return

    rate = len(session) / span_hours
    print(f"  Rate:               {rate:,.0f} videos/hour")
    if session_times:
        # Little's law: rate times service time is the average number in
        # flight, which is how many workers are actually doing something. Well
        # below --workers means they are waiting, not working.
        mean_seconds = sum(session_times) / len(session_times)
        print(f"  Mean per video:     {mean_seconds:.0f} s of worker time")
        print(f"  Workers busy:       {rate * mean_seconds / 3600.0:.1f} on average"
              " (compare with --workers)")


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
    finished: list[tuple[datetime, float | None]] = []

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

            # Rates count successes only. A failed video returns in under a
            # second, so mixing failures in inflates the rate while contributing
            # nothing to the mean worker time -- which is how the derived
            # parallelism came out above the worker count that produced it.
            if status != 'ok':
                continue
            # A malformed stamp costs that row from the rate, not the report.
            stamp = (row.get('completed_at') or '').strip()
            if not stamp:
                continue
            try:
                when = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            seconds = None
            with contextlib.suppress(ValueError):
                seconds = float(row['processing_time_seconds'])
            finished.append((when, seconds))

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

    print_throughput(finished)


if __name__ == '__main__':
    main()
