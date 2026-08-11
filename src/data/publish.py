"""Mirror the local output tree to an S3 prefix, and nothing else.

Results are written locally first and copied up; the local tree stays the source
of truth, so a failed or forbidden upload costs the copy, never the run's work.

This module is deliberately the only place that writes to S3, and it is
write-only in the narrowest sense: it calls ``put_object`` and ``upload_file``
and nothing else.  There is no delete, no copy, no bucket-level call and no
overwrite outside the destination prefix, because it shares a bucket with a
140,000-video corpus that must not be disturbed.  Two guards enforce that:

* the destination must name a non-empty prefix, so ``s3://bucket`` alone -- which
  would scatter results across the bucket root -- is refused rather than
  normalised into something plausible;
* every key is rebuilt from the prefix and re-checked against it before any
  request, so a relative path containing ``..`` cannot climb out of the
  destination and land on corpus data.

Write access is verified once, before a batch starts, for the same reason the
S3 reader verifies read access: a permission problem should stop the run at the
first request rather than surface 140,000 times.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from src.data.s3_source import S3AccessError, is_s3_uri, parse_s3_uri, s3_client

logger = logging.getLogger(__name__)

_UPLOAD_ERRORS = (ClientError, NoCredentialsError, BotoCoreError, OSError)

# Written once by verify_writable, and left in place: a probe that deleted itself
# would need delete permission this tool deliberately never asks for.
WRITE_CHECK_NAME = ".flicker_write_check"


def parse_destination(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` into its parts, requiring a real prefix.

    A destination without a prefix is refused: results would be written to the
    bucket root, which here is shared with the corpus itself.
    """
    if not is_s3_uri(uri):
        raise ValueError(f"S3 destination must be an s3://bucket/prefix URI: {uri!r}")
    location = parse_s3_uri(uri, require_key=False)
    prefix = location.key.strip("/")
    if not prefix:
        raise ValueError(
            f"S3 destination needs a prefix, not just a bucket: {uri!r}. "
            "Writing to the bucket root would mix results into the corpus."
        )
    if any(part in ("..", ".") for part in prefix.split("/")):
        raise ValueError(f"S3 destination prefix must not contain '.' or '..': {uri!r}")
    return location.bucket, prefix


@dataclass
class S3Publisher:
    """Copy files beneath a local root to the matching keys under an S3 prefix.

    ``dry_run`` resolves and logs every key without issuing a request, which is
    how a destination can be checked from a read-only role.
    """

    bucket: str
    prefix: str
    region: str | None = None
    profile: str | None = None
    dry_run: bool = False
    uploaded: int = field(default=0, init=False)
    failed: int = field(default=0, init=False)

    @classmethod
    def from_uri(
        cls,
        uri: str,
        *,
        region: str | None = None,
        profile: str | None = None,
        dry_run: bool = False,
    ) -> S3Publisher:
        bucket, prefix = parse_destination(uri)
        return cls(bucket=bucket, prefix=prefix, region=region, profile=profile, dry_run=dry_run)

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.prefix}/"

    def key_for(self, relative_path: str | Path) -> str:
        """Return the destination key for a path relative to the output root.

        The relative path is rebuilt component by component and the result is
        re-checked against the prefix, so a ``..`` segment cannot escape the
        destination into the corpus that shares this bucket.
        """
        parts = [
            part
            for part in PurePosixPath(str(relative_path).replace("\\", "/")).parts
            if part not in ("", "/", ".")
        ]
        # A component that is only whitespace would make a key nobody can address;
        # one *containing* a space is fine and common here, as
        # ``raw/Delhi_ZetWork/Night Shift/`` is a real prefix in this bucket.
        if not parts or any(part == ".." or not part.strip() for part in parts):
            raise ValueError(f"Unsafe relative path for an S3 key: {relative_path!r}")
        key = "/".join([self.prefix, *parts])
        if not key.startswith(f"{self.prefix}/"):
            raise ValueError(f"Refusing to write outside {self.prefix!r}: {key!r}")
        return key

    def verify_writable(self) -> None:
        """Confirm the destination accepts writes, before any work is done.

        Leaves a small marker object behind rather than deleting it: removing the
        probe would require delete permission, and this tool asks for none.
        """
        key = self.key_for(WRITE_CHECK_NAME)
        if self.dry_run:
            logger.info("[dry-run] would verify write access with s3://%s/%s", self.bucket, key)
            return
        body = b"Written by the flicker detector to verify write access.\n"
        try:
            self._client().put_object(Bucket=self.bucket, Key=key, Body=body)
        except _UPLOAD_ERRORS as error:
            raise S3AccessError(
                f"cannot write to {self.uri}: {error}\n"
                "The role needs s3:PutObject on "
                f"arn:aws:s3:::{self.bucket}/{self.prefix}/*. "
                "Re-run with --s3-output-dry-run to see what would be uploaded, "
                "or drop --s3-output to keep results local only."
            ) from error
        logger.info("Verified write access to %s", self.uri)

    def upload(self, local_path: str | Path, relative_path: str | Path) -> bool:
        """Copy one local file to its key.  Returns whether it was uploaded.

        A failure is counted and logged rather than raised: the local tree is the
        source of truth, and losing a copy must not lose the batch.  Access
        problems are caught up front by :meth:`verify_writable` instead.
        """
        local_path = Path(local_path)
        if not local_path.is_file():
            logger.warning("Nothing to upload at %s", local_path)
            return False
        key = self.key_for(relative_path)
        if self.dry_run:
            logger.info("[dry-run] s3://%s/%s <- %s", self.bucket, key, local_path)
            self.uploaded += 1
            return True
        try:
            self._client().upload_file(str(local_path), self.bucket, key)
        except _UPLOAD_ERRORS as error:
            logger.error(
                "Failed to upload %s to s3://%s/%s: %s", local_path, self.bucket, key, error
            )
            self.failed += 1
            return False
        logger.debug("Uploaded s3://%s/%s", self.bucket, key)
        self.uploaded += 1
        return True

    def upload_tree(self, root: str | Path, *, relative_to: str | Path | None = None) -> int:
        """Copy every file beneath ``root``, preserving its directory layout."""
        root = Path(root)
        base = Path(relative_to) if relative_to is not None else root
        count = 0
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if self.upload(path, path.relative_to(base)):
                count += 1
        return count

    def _client(self) -> Any:
        return s3_client(region=self.region, profile=self.profile)


def iter_output_files(root: str | Path, *, skip: tuple[str, ...] = ()) -> Iterator[Path]:
    """Yield the files of a local output tree, in a stable order."""
    root = Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path.name in skip:
            continue
        yield path
