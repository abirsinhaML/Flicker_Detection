#!/usr/bin/env python3
"""Convenience wrapper: run the detector on the pinned reference manifest.

Videos are read from S3, so AWS credentials must be present in the environment
(or a profile passed through as --aws-profile).  To score the corpus named by the
link sheet, call main.py with --links; to score a live bucket listing, --s3-prefix.

Three files are written: the durable per-window JSONL that --resume reads, the
flat per-window table for analysis, and the per-video rollup in the original
flag-manifest schema that calibration and the progress scripts read.

Usage:
    python run.py                       # process full manifest
    python run.py --limit 10            # first 10 videos only
    python run.py --workers 4 --resume  # resume with 4 parallel workers
"""

import subprocess
import sys

cmd = [
    sys.executable,
    "main.py",
    "--manifest",
    "data/manifest.csv",
    "--output",
    "output/window_metrics.jsonl",
    "--window-dir",
    "output/window_metrics",
    "--video-csv",
    "output/flag_manifest.csv",
    "--resume",
]
# Pass through any additional CLI arguments (e.g. --limit, --workers)
cmd.extend(sys.argv[1:])
raise SystemExit(subprocess.call(cmd))
