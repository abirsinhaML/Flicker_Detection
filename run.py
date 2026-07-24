#!/usr/bin/env python3
"""Convenience wrapper: run the detector on the reference manifest.

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
    "output/flag_manifest.csv",
    "--resume",
]
# Pass through any additional CLI arguments (e.g. --limit, --workers)
cmd.extend(sys.argv[1:])
raise SystemExit(subprocess.call(cmd))
