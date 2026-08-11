#!/usr/bin/env python3
"""How far along is a shard, and when will it finish?

Answers the question you actually have after detaching from tmux, which the
per-prefix report cannot: a shard knows how many videos it was given, so
progress is a fraction and the remaining time is arithmetic rather than a guess.

The denominator is the *resolvable* count, not the row count.  A shard of 23,171
rows may hold fewer real videos -- the link sheet names objects the bucket does
not have -- and that number is only known after the run resolves the links, so it
is read back from the run's own log.  Using the row count instead would leave
every shard stuck below 100%.

Rate is measured over the most recent contiguous session, not the whole file.  A
shard restarted with different settings, or after hours idle, would otherwise be
averaged with runs that no longer describe it -- and the estimate matters most
right after a change, which is exactly when the old numbers are most misleading.

Usage:
    python scripts/check_progress.py                     # every shard
    python scripts/check_progress.py output/shards/rows_0000000-0023170.jsonl
    python scripts/check_progress.py --watch              # refresh every 30s
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# A worker pool starts and finishes videos in clumps, so short gaps are normal.
# Only a longer one means the batch itself stopped and a new session began.
SESSION_GAP_SECONDS = 900
RESOLVED = re.compile(r"Link resolution: (\d+) of (\d+) links resolved")
ROW_RANGE = re.compile(r"rows (\d+)\.\.(\S+) inclusive: (\d+) of (\d+)")


def human(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def batch_running() -> bool:
    """Whether a scoring process is alive, by the same match stop_batch.sh uses."""
    try:
        listing = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="], capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return any(
        "main.py" in line
        and re.search(r"--(links|manifest|s3-prefix)", line)
        and "python" in line.split()[1]
        for line in listing.splitlines()
        if len(line.split()) > 2
    )


def shard_total(log_path: Path) -> tuple[int | None, str]:
    """Read the resolvable video count, and the row range, out of the run's log."""
    if not log_path.is_file():
        return None, ""
    resolved: int | None = None
    rows = ""
    for line in log_path.read_text(errors="replace").splitlines():
        found = RESOLVED.search(line)
        if found:
            resolved = int(found.group(1))
        found = ROW_RANGE.search(line)
        if found:
            rows = f"rows {found.group(1)}..{found.group(2)}"
    return resolved, rows


def report(records_path: Path, *, now: datetime) -> None:
    stamps: list[datetime] = []
    ok = errors = 0
    windows = 0
    for line in records_path.open(errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partial final line while the batch is mid-write
        if record.get("status") == "ok":
            ok += 1
            windows += len(record.get("windows") or ())
        else:
            errors += 1
        stamp = (record.get("completed_at") or "").strip()
        if stamp:
            # A malformed stamp costs that record from the rate, not the report.
            with contextlib.suppress(ValueError):
                stamps.append(datetime.fromisoformat(stamp))

    done = ok + errors
    total, rows = shard_total(records_path.with_suffix(".log"))
    print(f"\n\033[1m{records_path.stem}\033[0m  {rows}")

    if total:
        share = done / total * 100
        filled = int(share / 100 * 40)
        print(f"  [{'#' * filled}{'.' * (40 - filled)}] {share:5.1f}%")
        print(f"  scored     {done:,} of {total:,} resolvable   ({ok:,} ok, {errors:,} error)")
    else:
        print(f"  scored     {done:,}   ({ok:,} ok, {errors:,} error)")
        print("  (total unknown: the log has no 'Link resolution' line yet)")
    print(f"  windows    {windows:,}")

    if len(stamps) < 2:
        print("  rate       not enough completions yet\n")
        return

    stamps.sort()
    session = [stamps[0]]
    for previous, current in zip(stamps, stamps[1:], strict=False):
        if (current - previous).total_seconds() > SESSION_GAP_SECONDS:
            session = [current]
        else:
            session.append(current)

    span = (session[-1] - session[0]).total_seconds()
    if span <= 0:
        print("  rate       too early to measure\n")
        return

    per_hour = (len(session) - 1) / span * 3600
    earlier = len(stamps) - len(session)
    note = f" (+{earlier:,} from earlier sessions)" if earlier else ""
    print(f"  rate       {per_hour:,.0f} videos/hour over the last {human(span)}{note}")

    if total and per_hour > 0:
        remaining = total - done
        print(
            f"  remaining  {remaining:,} videos -> ~{human(remaining / per_hour * 3600)} "
            f"at this rate"
        )

    idle = (now - session[-1].astimezone(timezone.utc)).total_seconds()
    last = f"  last done  {session[-1].isoformat()}  ({human(idle)} ago)"
    # Long silence with a live process is normal on a 9-minute 4K video; long
    # silence with no process means the run stopped and nobody noticed.
    if idle > 1800:
        last += "  \033[33m<- quiet\033[0m"
    print(last + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records",
        nargs="*",
        help="Shard JSONL files (default: every one under output/shards)",
    )
    parser.add_argument("--watch", action="store_true", help="Refresh every 30 seconds")
    arguments = parser.parse_args()

    paths = [Path(path) for path in arguments.records] or sorted(
        Path("output/shards").glob("*.jsonl")
    )
    if not paths:
        print("No shard files found under output/shards/", file=sys.stderr)
        raise SystemExit(1)

    while True:
        if arguments.watch:
            print("\033[2J\033[H", end="")
        now = datetime.now(timezone.utc)
        print(f"{'batch process: RUNNING' if batch_running() else 'batch process: not running'}")
        for path in paths:
            if path.is_file():
                report(path, now=now)
            else:
                print(f"\n{path}: not found\n")
        if not arguments.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
