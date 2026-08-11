#!/usr/bin/env python3
"""Flatten the JSONL output into the window and video tables.

The batch already writes both CSVs as it runs.  This exists for the case where
it was told not to -- at corpus scale the window table is ~31 rows per video, so
a run may prefer to keep only the JSONL and materialize the flat table for the
slice it actually wants to analyse.

Usage:
    python scripts/export_windows.py output/window_metrics.jsonl
    python scripts/export_windows.py output/window_metrics.jsonl \\
        --windows output/windows.csv --videos output/videos.csv
    python scripts/export_windows.py output/window_metrics.jsonl \\
        --min-score 0.3 --band mild extreme --windows output/suspect_windows.csv
    python scripts/export_windows.py output/window_metrics.jsonl \\
        --window-dir output/window_metrics          # rebuild the per-video tree

Merge a sharded fleet's results into one rollup:
    python scripts/export_windows.py output/shards/*.jsonl \\
        --videos output/flag_manifest.csv
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.results import (  # noqa: E402
    read_records,
    write_video_csv,
    write_window_csv,
    write_window_tree,
)


def filtered(
    records: Iterable[dict[str, Any]],
    *,
    min_score: float | None,
    bands: tuple[str, ...],
    keys: tuple[str, ...],
) -> Iterator[dict[str, Any]]:
    """Drop windows that fail the filters, and videos left with none.

    Filtering here rather than after export is what makes a corpus-scale JSONL
    usable: the interesting slice is normally a few percent of the windows, and a
    dataframe of the rest is only there to be discarded.
    """
    for record in records:
        if keys and record.get("video_key") not in keys:
            continue
        if min_score is None and not bands:
            yield record
            continue
        windows = [
            window
            for window in record.get("windows") or []
            if (min_score is None or float(window.get("score", 0.0)) >= min_score)
            and (not bands or window.get("severity_band") in bands)
        ]
        if not windows:
            continue
        yield {**record, "windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records",
        nargs="+",
        help=(
            "One or more JSONL outputs written by main.py --output. Several are "
            "read in order, which is how a sharded fleet's results are merged."
        ),
    )
    parser.add_argument(
        "--windows", help="Write every window into one combined CSV at this path"
    )
    parser.add_argument(
        "--window-dir",
        help="Rebuild the per-video window CSV tree under this root, one file per video",
    )
    parser.add_argument("--videos", help="Write the one-row-per-video rollup here")
    parser.add_argument(
        "--min-score",
        type=float,
        help="Keep only windows scoring at or above this",
    )
    parser.add_argument(
        "--band",
        nargs="+",
        default=[],
        choices=["none", "mild", "extreme"],
        help="Keep only windows in these severity bands",
    )
    parser.add_argument(
        "--key",
        nargs="+",
        default=[],
        help="Keep only these video keys",
    )
    arguments = parser.parse_args()
    if not (arguments.windows or arguments.videos or arguments.window_dir):
        parser.error("nothing to do: pass --windows, --window-dir, and/or --videos")

    records_paths = [Path(path) for path in arguments.records]
    for path in records_paths:
        if not path.exists():
            parser.error(f"no such file: {path}")

    # Read once per output rather than holding the corpus in memory.  Several
    # shard files are chained, so a fleet's results merge without a separate step.
    def selected() -> Iterator[dict[str, Any]]:
        return filtered(
            chain.from_iterable(read_records(path) for path in records_paths),
            min_score=arguments.min_score,
            bands=tuple(arguments.band),
            keys=tuple(arguments.key),
        )

    if arguments.windows:
        written = write_window_csv(arguments.windows, selected())
        print(f"Wrote {written:,} window rows to {arguments.windows}")
    if arguments.window_dir:
        written = write_window_tree(arguments.window_dir, selected())
        print(f"Wrote {written:,} per-video window CSVs under {arguments.window_dir}")
    if arguments.videos:
        written = write_video_csv(arguments.videos, selected())
        print(f"Wrote {written:,} video rows to {arguments.videos}")


if __name__ == "__main__":
    main()
