#!/usr/bin/env python3
"""Upload everything under output/ to s3://stage-egocentric-humyn-data/flicker_results/

Standalone: needs only boto3 and imports nothing from this repo, so it can be
copied to any machine that has the results and run on its own.

The local tree's layout is reproduced verbatim beneath the prefix:

    output/shards/rows_0000000-0023170.jsonl
      -> s3://stage-egocentric-humyn-data/flicker_results/shards/rows_0000000-0023170.jsonl
    output/window_metrics/raw/Delhi_ZetWork/2026-06-20/DV0051/GX010085.csv
      -> s3://.../flicker_results/window_metrics/raw/Delhi_ZetWork/2026-06-20/DV0051/GX010085.csv

This only ever PUTs objects under the destination prefix.  It never deletes, never
copies, and never writes outside the prefix -- every key is rebuilt from the
prefix and re-checked before the request, so a path containing ".." cannot climb
out.  That matters because these buckets also hold source data.

Usage:
    python upload_output.py                      # upload output/ as configured
    python upload_output.py --dry-run            # print every key, upload nothing
    python upload_output.py --skip-existing      # skip objects already the same size
    python upload_output.py --skip detector.log  # leave named files behind
    python upload_output.py --source some/dir --dest s3://bucket/prefix/
    python upload_output.py --region us-east-1   # if the bucket is not in ap-south-1

Exit codes: 0 all uploaded, 1 some failed, 2 could not start (bad path/credentials).
"""

from __future__ import annotations

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:  # pragma: no cover - environment problem, not a code path
    sys.exit("boto3 is required:  pip install boto3")

DEFAULT_SOURCE = "output"
DEFAULT_DEST = "s3://stage-egocentric-humyn-data/flicker_results/"
DEFAULT_REGION = "ap-south-1"

# What S3 returns when the client signed for the wrong region.  None of the codes
# mention regions, so the hint below has to name the cause explicitly.
REGION_ERRORS = {
    "PermanentRedirect",
    "AuthorizationHeaderMalformed",
    "IllegalLocationConstraintException",
}

_print_lock = threading.Lock()


def parse_destination(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` into its parts, requiring a real prefix.

    A destination without a prefix is refused rather than treated as the bucket
    root: these buckets hold source data, and results scattered across the root
    would be mixed into it.
    """
    if not uri.lower().startswith("s3://"):
        raise ValueError(f"destination must be an s3://bucket/prefix URI: {uri!r}")
    remainder = uri[5:].strip("/")
    bucket, _, prefix = remainder.partition("/")
    if not bucket:
        raise ValueError(f"destination is missing a bucket: {uri!r}")
    prefix = prefix.strip("/")
    if not prefix:
        raise ValueError(
            f"destination needs a prefix, not just a bucket: {uri!r}\n"
            "Writing to the bucket root would mix results into the source data."
        )
    if any(part in (".", "..") for part in prefix.split("/")):
        raise ValueError(f"destination prefix must not contain '.' or '..': {uri!r}")
    return bucket, prefix


def key_for(prefix: str, relative_path: Path) -> str:
    """Return the destination key for a path relative to the source root.

    Rebuilt component by component and re-checked against the prefix, so a
    ``..`` segment cannot escape the destination.
    """
    parts = [
        part
        for part in PurePosixPath(relative_path.as_posix()).parts
        if part not in ("", "/", ".")
    ]
    # A component that is only whitespace makes a key nobody can address.  One
    # *containing* a space is fine: "Night Shift" is a real directory here.
    if not parts or any(part == ".." or not part.strip() for part in parts):
        raise ValueError(f"unsafe relative path for a key: {relative_path!r}")
    key = "/".join([prefix, *parts])
    if not key.startswith(f"{prefix}/"):
        raise ValueError(f"refusing to write outside {prefix!r}: {key!r}")
    return key


def write_hint(bucket: str, prefix: str, region: str, error: Exception) -> str:
    """Name the likely cause: a wrong region and a missing grant look alike."""
    code = ""
    if isinstance(error, ClientError):
        code = str(error.response.get("Error", {}).get("Code", ""))
    if code in REGION_ERRORS:
        return (
            f"This looks like a region mismatch: the client signed for {region!r}.\n"
            f"Find the bucket's region with:\n"
            f"  aws s3api get-bucket-location --bucket {bucket}\n"
            f"then re-run with --region <that region>."
        )
    if isinstance(error, NoCredentialsError) or code in ("InvalidAccessKeyId", "ExpiredToken"):
        return (
            "No usable credentials. Export them and retry:\n"
            '  export AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." '
            'AWS_SESSION_TOKEN="..."'
        )
    return (
        f"The role needs s3:PutObject on arn:aws:s3:::{bucket}/{prefix}/*\n"
        "Run with --dry-run to see what would be uploaded without sending anything."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE, help=f"default: {DEFAULT_SOURCE}")
    parser.add_argument("--dest", default=DEFAULT_DEST, help=f"default: {DEFAULT_DEST}")
    parser.add_argument("--region", default=DEFAULT_REGION, help=f"default: {DEFAULT_REGION}")
    parser.add_argument("--profile", help="AWS named profile (default: standard chain)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every destination key without sending anything",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose object already exists with the same byte size",
    )
    parser.add_argument("--skip", nargs="+", default=[], help="File names to leave behind")
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Parallel uploads; these are many small files, so this matters (default: 8)",
    )
    arguments = parser.parse_args()

    source = Path(arguments.source)
    if not source.is_dir():
        print(f"not a directory: {source}", file=sys.stderr)
        return 2
    try:
        bucket, prefix = parse_destination(arguments.dest)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    skip = set(arguments.skip)
    files = [
        path
        for path in sorted(p for p in source.rglob("*") if p.is_file())
        if path.name not in skip
    ]
    if not files:
        print(f"nothing to upload under {source}/")
        return 0

    total_bytes = sum(path.stat().st_size for path in files)
    print(f"{len(files):,} files, {total_bytes / 1e6:,.1f} MB")
    print(f"  from  {source}/")
    print(f"  to    s3://{bucket}/{prefix}/")
    if arguments.dry_run:
        print("  (dry run: nothing will be sent)")
    print()

    session = boto3.Session(profile_name=arguments.profile, region_name=arguments.region)
    client = session.client("s3")

    # One PutObject before fanning out, so a permission or region problem stops
    # here with an explanation instead of failing once per file.
    if not arguments.dry_run:
        try:
            client.put_object(
                Bucket=bucket,
                Key=f"{prefix}/.upload_check",
                Body=b"Written to verify write access.\n",
            )
        except (ClientError, BotoCoreError, NoCredentialsError) as error:
            print(f"cannot write to s3://{bucket}/{prefix}/: {error}\n", file=sys.stderr)
            print(write_hint(bucket, prefix, arguments.region, error), file=sys.stderr)
            return 2

    counts = {"uploaded": 0, "skipped": 0, "failed": 0}

    def transfer(path: Path) -> tuple[str, str | None]:
        """Upload one file.  Returns (outcome, message)."""
        relative = path.relative_to(source)
        try:
            key = key_for(prefix, relative)
        except ValueError as error:
            return "failed", str(error)

        if arguments.dry_run:
            return "uploaded", f"s3://{bucket}/{key}"

        if arguments.skip_existing:
            try:
                head = client.head_object(Bucket=bucket, Key=key)
                if head["ContentLength"] == path.stat().st_size:
                    return "skipped", key
            except ClientError as error:
                # 404 simply means it is not there yet; anything else is worth
                # surfacing rather than silently re-uploading.
                if error.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
                    return "failed", f"{key}: {error}"
            except (BotoCoreError, OSError) as error:
                return "failed", f"{key}: {error}"

        try:
            client.upload_file(str(path), bucket, key)
        except (ClientError, BotoCoreError, NoCredentialsError, OSError) as error:
            return "failed", f"{key}: {error}"
        return "uploaded", key

    with ThreadPoolExecutor(max_workers=max(1, arguments.threads)) as pool:
        futures = {pool.submit(transfer, path): path for path in files}
        for finished, future in enumerate(as_completed(futures), start=1):
            outcome, message = future.result()
            counts[outcome] += 1
            with _print_lock:
                if outcome == "failed":
                    print(f"  FAILED  {message}", file=sys.stderr)
                elif arguments.dry_run:
                    print(f"  would upload  {message}")
                elif finished % 200 == 0 or finished == len(files):
                    print(
                        f"  {finished:,}/{len(files):,}  "
                        f"uploaded={counts['uploaded']:,} "
                        f"skipped={counts['skipped']:,} failed={counts['failed']:,}"
                    )

    verb = "would upload" if arguments.dry_run else "uploaded"
    print(
        f"\n{verb} {counts['uploaded']:,}, skipped {counts['skipped']:,}, "
        f"failed {counts['failed']:,}"
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
