"""Coverage for S3 listing, just-in-time signing, and URI handling."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

from botocore.exceptions import ClientError, NoCredentialsError

from src.data.s3_source import (
    S3AccessError,
    S3Settings,
    S3VideoCatalog,
    SourceResolver,
    build_s3_uri,
    is_s3_uri,
    parse_s3_uri,
    redact_url,
)

BUCKET = "humyn-data-partners-prod"
PREFIX = "visionlab/visionlab/outbound/"


class FakePaginator:
    def __init__(self, pages: list[dict[str, Any]]) -> None:
        self.pages = pages
        self.calls: list[dict[str, Any]] = []

    def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return self.pages


class FakeS3Client:
    """Stands in for a boto3 S3 client without touching the network."""

    def __init__(
        self,
        pages: list[dict[str, Any]] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.paginator = FakePaginator(pages or [])
        self.error = error
        self.presign_calls: list[dict[str, Any]] = []
        self.head_calls: list[dict[str, Any]] = []

    def get_paginator(self, name: str) -> FakePaginator:
        if self.error is not None:
            raise self.error
        return self.paginator

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        self.head_calls.append(kwargs)
        return {"ContentLength": 1}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        contents = [item for page in self.paginator.pages for item in page.get("Contents", ())]
        return {"KeyCount": len(contents[: int(kwargs.get("MaxKeys", 1000))])}

    def generate_presigned_url(self, operation: str, **kwargs: Any) -> str:
        if self.error is not None:
            raise self.error
        self.presign_calls.append(kwargs)
        params = kwargs["Params"]
        return f"https://{params['Bucket']}.s3.amazonaws.com/{params['Key']}?X-Amz-Signature=abc"


def _page(*objects: tuple[str, int]) -> dict[str, Any]:
    return {
        "Contents": [
            {
                "Key": key,
                "Size": size,
                "LastModified": datetime(2026, 6, 13, 17, 35, 52, tzinfo=timezone.utc),
            }
            for key, size in objects
        ]
    }


class UriTests(unittest.TestCase):
    def test_parses_bucket_and_nested_key(self) -> None:
        location = parse_s3_uri(f"s3://{BUCKET}/{PREFIX}India_Ahmedabad_001.mp4")
        self.assertEqual(location.bucket, BUCKET)
        self.assertEqual(location.key, f"{PREFIX}India_Ahmedabad_001.mp4")
        self.assertEqual(location.uri, f"s3://{BUCKET}/{PREFIX}India_Ahmedabad_001.mp4")

    def test_allows_prefix_uri_only_when_key_not_required(self) -> None:
        self.assertEqual(parse_s3_uri(f"s3://{BUCKET}/{PREFIX}", require_key=False).key, PREFIX)
        with self.assertRaises(ValueError):
            parse_s3_uri(f"s3://{BUCKET}/")

    def test_rejects_non_s3_and_bucketless_uris(self) -> None:
        with self.assertRaises(ValueError):
            parse_s3_uri("https://example.com/a.mp4")
        with self.assertRaises(ValueError):
            parse_s3_uri("s3:///key.mp4", require_key=False)

    def test_recognizes_and_builds_uris(self) -> None:
        self.assertTrue(is_s3_uri(f"s3://{BUCKET}/a.mp4"))
        self.assertTrue(is_s3_uri(f"  S3://{BUCKET}/a.mp4  "))
        self.assertFalse(is_s3_uri("/tmp/a.mp4"))
        self.assertEqual(build_s3_uri(BUCKET, "/a.mp4"), f"s3://{BUCKET}/a.mp4")

    def test_redacts_signature_but_keeps_object_path(self) -> None:
        redacted = redact_url("https://b.s3.amazonaws.com/a.mp4?X-Amz-Signature=secret")
        self.assertNotIn("secret", redacted)
        self.assertIn("a.mp4", redacted)
        self.assertEqual(redact_url("/tmp/a.mp4"), "/tmp/a.mp4")


class CatalogTests(unittest.TestCase):
    def test_lists_only_video_objects_as_durable_uris(self) -> None:
        client = FakeS3Client(
            [
                _page((f"{PREFIX}a.mp4", 100), (f"{PREFIX}notes.txt", 50)),
                # Zero-byte folder placeholders are not videos.
                _page((PREFIX, 0), (f"{PREFIX}b.MP4", 200)),
            ]
        )
        with patch("src.data.s3_source.s3_client", return_value=client):
            entries = list(S3VideoCatalog(BUCKET, PREFIX).list_entries())

        self.assertEqual([entry.key for entry in entries], [f"{PREFIX}a.mp4", f"{PREFIX}b.MP4"])
        self.assertEqual(entries[0].source_uri, f"s3://{BUCKET}/{PREFIX}a.mp4")
        self.assertEqual(entries[0].size_bytes, 100)
        self.assertEqual(
            entries[0].last_modified,
            datetime(2026, 6, 13, 17, 35, 52, tzinfo=timezone.utc),
        )
        self.assertEqual(client.paginator.calls[0], {"Bucket": BUCKET, "Prefix": PREFIX})

    def test_limit_stops_listing_early(self) -> None:
        client = FakeS3Client([_page((f"{PREFIX}a.mp4", 1), (f"{PREFIX}b.mp4", 1))])
        with patch("src.data.s3_source.s3_client", return_value=client):
            entries = list(S3VideoCatalog(BUCKET, PREFIX).list_entries(limit=1))
        self.assertEqual(len(entries), 1)

    def test_rejects_non_positive_limit(self) -> None:
        with (
            patch("src.data.s3_source.s3_client", return_value=FakeS3Client()),
            self.assertRaises(ValueError),
        ):
            list(S3VideoCatalog(BUCKET, PREFIX).list_entries(limit=0))

    def test_missing_credentials_surface_as_access_error(self) -> None:
        client = FakeS3Client([_page((f"{PREFIX}a.mp4", 1))], error=NoCredentialsError())
        with patch("src.data.s3_source.s3_client", return_value=client):
            catalog = S3VideoCatalog(BUCKET, PREFIX)
            with self.assertRaises(S3AccessError) as caught:
                list(catalog.list_entries())
            with self.assertRaises(S3AccessError):
                catalog.verify_access()
        self.assertIn("AWS_SESSION_TOKEN", str(caught.exception))

    def test_verify_access_rejects_empty_prefix(self) -> None:
        with (
            patch("src.data.s3_source.s3_client", return_value=FakeS3Client([])),
            self.assertRaises(S3AccessError),
        ):
            S3VideoCatalog(BUCKET, PREFIX).verify_access()

    def test_verify_access_accepts_populated_prefix(self) -> None:
        with patch(
            "src.data.s3_source.s3_client",
            return_value=FakeS3Client([_page((f"{PREFIX}a.mp4", 1))]),
        ):
            S3VideoCatalog(BUCKET, PREFIX).verify_access()

    def test_requires_a_bucket(self) -> None:
        with self.assertRaises(ValueError):
            S3VideoCatalog("")


class ResolverTests(unittest.TestCase):
    def test_passes_through_local_paths_and_plain_urls(self) -> None:
        resolver = SourceResolver()
        self.assertEqual(resolver.resolve("/tmp/sample.mp4"), "/tmp/sample.mp4")
        self.assertEqual(resolver.resolve("https://host/a.mp4"), "https://host/a.mp4")

    def test_signs_s3_uris_with_configured_expiry(self) -> None:
        client = FakeS3Client()
        with patch("src.data.s3_source.s3_client", return_value=client):
            url = SourceResolver(presign_expiry=900).resolve(f"s3://{BUCKET}/{PREFIX}a.mp4")
        self.assertTrue(url.startswith("https://"))
        self.assertEqual(client.presign_calls[0]["ExpiresIn"], 900)
        self.assertEqual(
            client.presign_calls[0]["Params"],
            {"Bucket": BUCKET, "Key": f"{PREFIX}a.mp4"},
        )

    def test_verify_readable_heads_s3_objects_and_skips_others(self) -> None:
        client = FakeS3Client()
        with patch("src.data.s3_source.s3_client", return_value=client):
            SourceResolver().verify_readable(f"s3://{BUCKET}/{PREFIX}a.mp4")
            SourceResolver().verify_readable("/tmp/local.mp4")
        self.assertEqual(client.head_calls, [{"Bucket": BUCKET, "Key": f"{PREFIX}a.mp4"}])

    def test_verify_readable_reports_unreadable_objects(self) -> None:
        error = ClientError({"Error": {"Code": "ExpiredToken"}}, "HeadObject")
        with (
            patch("src.data.s3_source.s3_client", return_value=FakeS3Client(error=error)),
            self.assertRaises(S3AccessError),
        ):
            SourceResolver().verify_readable(f"s3://{BUCKET}/{PREFIX}a.mp4")

    def test_signing_without_credentials_surfaces_access_error(self) -> None:
        with (
            patch(
                "src.data.s3_source.s3_client",
                return_value=FakeS3Client(error=NoCredentialsError()),
            ),
            self.assertRaises(S3AccessError),
        ):
            SourceResolver().resolve(f"s3://{BUCKET}/a.mp4")


class SettingsTests(unittest.TestCase):
    def test_reads_configured_block(self) -> None:
        settings = S3Settings.from_config(
            {
                "s3": {
                    "bucket": BUCKET,
                    "prefix": PREFIX,
                    "region": "ap-south-1",
                    "presign_expiry_seconds": 600,
                    "extensions": [".mp4", ".mov"],
                }
            }
        )
        self.assertEqual(settings.bucket, BUCKET)
        self.assertEqual(settings.extensions, (".mp4", ".mov"))
        self.assertEqual(settings.resolver().presign_expiry, 600)

    def test_defaults_when_block_absent(self) -> None:
        settings = S3Settings.from_config({})
        self.assertIsNone(settings.bucket)
        self.assertEqual(settings.region, "ap-south-1")
        self.assertEqual(settings.extensions, (".mp4",))

    def test_catalog_honors_listing_root_forms(self) -> None:
        settings = S3Settings.from_config({"s3": {"bucket": BUCKET, "prefix": PREFIX}})

        configured = settings.catalog()
        self.assertEqual((configured.bucket, configured.prefix), (BUCKET, PREFIX))

        bare_prefix = settings.catalog("other/prefix/")
        self.assertEqual((bare_prefix.bucket, bare_prefix.prefix), (BUCKET, "other/prefix/"))

        full_uri = settings.catalog("s3://another-bucket/deep/prefix/")
        self.assertEqual((full_uri.bucket, full_uri.prefix), ("another-bucket", "deep/prefix/"))

    def test_catalog_without_any_bucket_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            S3Settings.from_config({}).catalog()


if __name__ == "__main__":
    unittest.main()
