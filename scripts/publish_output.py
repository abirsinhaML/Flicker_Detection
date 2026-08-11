#!/usr/bin/env python3
"""Copy an existing local output tree to its S3 prefix.

The batch uploads as it runs when given ``--s3-output``.  This covers the other
cases: results scored before publishing existed, a run whose uploads were denied
at the time, or a re-copy after the tree was rebuilt with
``scripts/export_windows.py``.

Nothing is deleted and nothing outside the destination prefix is written -- see
``src/data/publish.py``.  Files already present in S3 are overwritten, because
the local tree is the source of truth.

Usage:
    python scripts/publish_output.py output \\
        s3://prod-egocentric-humyn-data/raw/flicker_result/

    python scripts/publish_output.py output s3://bucket/raw/flicker_result/ \\
        --dry-run                      # resolve every key, upload nothing
    python scripts/publish_output.py output s3://bucket/raw/flicker_result/ \\
        --skip detector.log flag_manifest.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.publish import S3Publisher  # noqa: E402
from src.data.s3_source import S3AccessError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Local output directory to copy")
    parser.add_argument("destination", help="s3://bucket/prefix to copy it to")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print every destination key without uploading",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="File names to leave behind, e.g. detector.log",
    )
    parser.add_argument("--aws-region", default="ap-south-1")
    parser.add_argument("--aws-profile")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    root = Path(arguments.root)
    if not root.is_dir():
        parser.error(f"not a directory: {root}")

    publisher = S3Publisher.from_uri(
        arguments.destination,
        region=arguments.aws_region,
        profile=arguments.aws_profile,
        dry_run=arguments.dry_run,
    )
    files = [
        path
        for path in sorted(p for p in root.rglob("*") if p.is_file())
        if path.name not in set(arguments.skip)
    ]
    if not files:
        print(f"Nothing to publish under {root}")
        return

    print(f"Publishing {len(files)} files from {root} to {publisher.uri}")
    try:
        publisher.verify_writable()
    except S3AccessError as error:
        print(f"\n{error}", file=sys.stderr)
        raise SystemExit(2) from error

    for path in files:
        publisher.upload(path, path.relative_to(root))

    verb = "would upload" if arguments.dry_run else "uploaded"
    print(f"\n{verb} {publisher.uploaded} files, {publisher.failed} failed")
    if publisher.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
