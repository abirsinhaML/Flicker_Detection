"""Durable S3 access to the video corpus.

Presigned URLs expire, so the pipeline never persists one.  Manifests carry only
``s3://bucket/key`` URIs; each worker signs a short-lived URL for its own key
immediately before decoding it, using whatever credentials the standard AWS
chain supplies (environment variables, a named profile, or an instance role).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from src.core.types import ManifestEntry

logger = logging.getLogger(__name__)

S3_SCHEME = "s3"
DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4",)
DEFAULT_PRESIGN_EXPIRY = 3600
DEFAULT_REGION = "ap-south-1"

_CREDENTIAL_ERRORS = (ClientError, NoCredentialsError, BotoCoreError)


class S3AccessError(RuntimeError):
    """Raised when S3 is unreachable or credentials are missing or expired."""


@dataclass(frozen=True, slots=True)
class S3Location:
    """A parsed ``s3://bucket/key`` reference."""

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"{S3_SCHEME}://{self.bucket}/{self.key}"


def is_s3_uri(value: str) -> bool:
    """Return whether ``value`` is an ``s3://`` URI."""
    return str(value).strip().lower().startswith(f"{S3_SCHEME}://")


def parse_s3_uri(uri: str, *, require_key: bool = True) -> S3Location:
    """Parse ``s3://bucket/key`` into its bucket and key.

    ``require_key=False`` permits a bare bucket or prefix URI, which is how a
    listing root such as ``s3://bucket/some/prefix/`` is expressed.
    """
    text = str(uri).strip()
    if not is_s3_uri(text):
        raise ValueError(f"Not an s3:// URI: {uri}")

    parsed = urlparse(text)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket:
        raise ValueError(f"S3 URI is missing a bucket: {uri}")
    if require_key and not key:
        raise ValueError(f"S3 URI is missing an object key: {uri}")
    return S3Location(bucket=bucket, key=key)


def build_s3_uri(bucket: str, key: str) -> str:
    """Compose an ``s3://bucket/key`` URI."""
    return f"{S3_SCHEME}://{bucket}/{key.lstrip('/')}"


def redact_url(value: str) -> str:
    """Drop the query string so signed URLs never reach logs or output files.

    A presigned URL's signature is a bearer credential for the object; logging
    one verbatim would leak read access for the lifetime of the signature.
    """
    text = str(value)
    if "?" not in text:
        return text
    return f"{text.split('?', 1)[0]}?<signature-redacted>"


# A URL embedded in surrounding prose, up to the first whitespace or quote.
_EMBEDDED_URL = re.compile(r"(https?://[^\s'\"]+?)\?[^\s'\"]*")


def redact_text(value: str) -> str:
    """Strip the query string from every URL inside a larger piece of text.

    FFmpeg reports a failed open by quoting the whole URL back, so an error
    message is where a signed URL escapes: unlike a log line, that message is
    persisted, as the ``error`` field of the video's output record.  The
    signature is a bearer credential for the object, and one written to an
    output file outlives the run that produced it.

    Distinct from :func:`redact_url` because this must not truncate at the first
    ``?`` -- that character occurs in ordinary error prose, and cutting there
    would discard the part of the message worth keeping.
    """
    return _EMBEDDED_URL.sub(r"\1?<signature-redacted>", str(value))


# Cached per process.  boto3 clients are neither picklable nor safe to share
# across a fork, so every worker builds and caches its own on first use.
_CLIENT_CACHE: dict[tuple[str | None, str | None], Any] = {}


def s3_client(*, region: str | None = None, profile: str | None = None) -> Any:
    """Return this process's cached S3 client for ``region`` and ``profile``."""
    cache_key = (region, profile)
    client = _CLIENT_CACHE.get(cache_key)
    if client is None:
        session = boto3.Session(profile_name=profile, region_name=region)
        client = session.client(
            "s3",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "adaptive"},
                max_pool_connections=32,
            ),
        )
        _CLIENT_CACHE[cache_key] = client
    return client


@dataclass(frozen=True, slots=True)
class SourceResolver:
    """Turn a durable ``source_uri`` into a URI FFmpeg can open right now.

    Holds only configuration, so it is picklable and can be handed to worker
    processes.  Each worker signs its own URLs from its own cached client.
    """

    region: str | None = DEFAULT_REGION
    profile: str | None = None
    presign_expiry: int = DEFAULT_PRESIGN_EXPIRY

    def resolve(self, source_uri: str) -> str:
        """Sign an ``s3://`` URI; pass local paths and plain URLs through."""
        if not is_s3_uri(source_uri):
            return source_uri
        return self.presign(parse_s3_uri(source_uri))

    def verify_readable(self, source_uri: str) -> None:
        """Confirm one S3 object is actually readable; no-op for other sources.

        Signing is offline, so it succeeds even with expired credentials and the
        failure only surfaces when FFmpeg gets a 403.  One HEAD in the parent
        checks credentials, region, and object permissions together, instead of
        letting every worker rediscover the same failure per video.
        """
        if not is_s3_uri(source_uri):
            return
        location = parse_s3_uri(source_uri)
        client = s3_client(region=self.region, profile=self.profile)
        try:
            client.head_object(Bucket=location.bucket, Key=location.key)
        except _CREDENTIAL_ERRORS as error:
            raise S3AccessError(_credential_hint(f"cannot read {location.uri}", error)) from error

    def presign(self, location: S3Location, *, expires_in: int | None = None) -> str:
        """Return a short-lived HTTPS GET URL for one object."""
        client = s3_client(region=self.region, profile=self.profile)
        try:
            return str(
                client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": location.bucket, "Key": location.key},
                    ExpiresIn=int(expires_in or self.presign_expiry),
                )
            )
        except _CREDENTIAL_ERRORS as error:
            raise S3AccessError(_credential_hint(f"cannot sign {location.uri}", error)) from error


class S3VideoCatalog:
    """Enumerate the video objects under one bucket prefix."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        region: str | None = DEFAULT_REGION,
        profile: str | None = None,
        extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
        presign_expiry: int = DEFAULT_PRESIGN_EXPIRY,
    ) -> None:
        if not bucket:
            raise ValueError("bucket must be a non-empty name")
        self.bucket = bucket
        self.prefix = prefix.lstrip("/")
        self.extensions = tuple(extension.lower() for extension in extensions)
        self.resolver = SourceResolver(
            region=region,
            profile=profile,
            presign_expiry=presign_expiry,
        )

    @property
    def client(self) -> Any:
        return s3_client(region=self.resolver.region, profile=self.resolver.profile)

    def verify_access(self) -> None:
        """Fail fast, in the parent, on bad credentials, region, or prefix.

        Worth one request before fanning out: otherwise every worker rediscovers
        the same authentication failure and reports it as a decode error.
        """
        try:
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix,
                MaxKeys=1,
            )
        except _CREDENTIAL_ERRORS as error:
            raise S3AccessError(
                _credential_hint(f"cannot list s3://{self.bucket}/{self.prefix}", error)
            ) from error
        if not response.get("KeyCount"):
            raise S3AccessError(f"No objects found under s3://{self.bucket}/{self.prefix}")

    def list_entries(self, limit: int | None = None) -> Iterator[ManifestEntry]:
        """Yield one entry per video object under the prefix, in listing order.

        Streams the paginated listing rather than materializing it, so a prefix
        holding hundreds of thousands of keys costs constant memory.
        """
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero")

        yielded = 0
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", ()):
                    key = str(obj["Key"])
                    size = int(obj.get("Size", 0))
                    # Zero-byte keys ending in "/" are console-created folder
                    # placeholders, not videos.
                    if size == 0 or not self._is_video(key):
                        continue
                    yield ManifestEntry(
                        key=key,
                        source_uri=build_s3_uri(self.bucket, key),
                        size_bytes=size,
                        last_modified=obj.get("LastModified"),
                    )
                    yielded += 1
                    if limit is not None and yielded >= limit:
                        return
        except _CREDENTIAL_ERRORS as error:
            raise S3AccessError(
                _credential_hint(f"cannot list s3://{self.bucket}/{self.prefix}", error)
            ) from error

    def presign(self, key: str, *, expires_in: int | None = None) -> str:
        """Return a short-lived HTTPS GET URL for one key in this bucket."""
        return self.resolver.presign(
            S3Location(bucket=self.bucket, key=key),
            expires_in=expires_in,
        )

    def _is_video(self, key: str) -> bool:
        return not self.extensions or key.lower().endswith(self.extensions)


@dataclass(frozen=True, slots=True)
class S3Settings:
    """The ``s3:`` block of the detector configuration."""

    bucket: str | None = None
    prefix: str = ""
    region: str | None = DEFAULT_REGION
    profile: str | None = None
    presign_expiry_seconds: int = DEFAULT_PRESIGN_EXPIRY
    extensions: tuple[str, ...] = field(default=DEFAULT_VIDEO_EXTENSIONS)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> S3Settings:
        """Read settings from a loaded detector config, tolerating absence."""
        section = config.get("s3") or {}
        extensions = section.get("extensions") or DEFAULT_VIDEO_EXTENSIONS
        return cls(
            bucket=section.get("bucket"),
            prefix=str(section.get("prefix") or ""),
            region=section.get("region") or DEFAULT_REGION,
            profile=section.get("profile"),
            presign_expiry_seconds=int(
                section.get("presign_expiry_seconds") or DEFAULT_PRESIGN_EXPIRY
            ),
            extensions=tuple(str(extension) for extension in extensions),
        )

    def resolver(self) -> SourceResolver:
        """Build the just-in-time URL signer these settings describe."""
        return SourceResolver(
            region=self.region,
            profile=self.profile,
            presign_expiry=self.presign_expiry_seconds,
        )

    def catalog(self, listing_root: str | None = None) -> S3VideoCatalog:
        """Build a catalog for ``listing_root`` or the configured bucket prefix.

        ``listing_root`` is an ``s3://bucket/prefix`` URI or a bare prefix to be
        read from the configured bucket.
        """
        bucket, prefix = self.bucket, self.prefix
        if listing_root:
            if is_s3_uri(listing_root):
                location = parse_s3_uri(listing_root, require_key=False)
                bucket, prefix = location.bucket, location.key
            else:
                prefix = listing_root.lstrip("/")
        if not bucket:
            raise ValueError(
                "No S3 bucket configured. Pass a full s3://bucket/prefix URI, "
                "or set s3.bucket in the detector config."
            )
        return S3VideoCatalog(
            bucket,
            prefix,
            region=self.region,
            profile=self.profile,
            extensions=self.extensions,
            presign_expiry=self.presign_expiry_seconds,
        )


def _credential_hint(action: str, error: Exception) -> str:
    return (
        f"{action}: {error}\n"
        "Verify that unexpired AWS credentials are visible to this process:\n"
        '  export AWS_ACCESS_KEY_ID="..." AWS_SECRET_ACCESS_KEY="..." '
        'AWS_SESSION_TOKEN="..."\n'
        "Temporary ASIA... credentials are short-lived; re-issue them and rerun "
        "with --resume to continue where the batch stopped."
    )
