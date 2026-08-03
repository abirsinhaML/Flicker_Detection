"""Coverage for reading and writing manifests of durable video identifiers."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.core.types import ManifestEntry
from src.data.manifest import MANIFEST_FIELDS, ManifestReader, key_for_uri, write_manifest

BUCKET = "humyn-data-partners-prod"
PREFIX = "visionlab/visionlab/outbound/"


class ManifestReaderTests(unittest.TestCase):
    def test_reads_source_uri_column_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            _write_csv(
                path,
                ("key", "source_uri", "size_bytes", "last_modified"),
                [
                    {
                        "key": f"{PREFIX}a.mp4",
                        "source_uri": f"s3://{BUCKET}/{PREFIX}a.mp4",
                        "size_bytes": "569135399",
                        "last_modified": "2026-06-13 17:35:52",
                    }
                ],
            )
            entries = list(ManifestReader(path))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].key, f"{PREFIX}a.mp4")
        self.assertEqual(entries[0].source_uri, f"s3://{BUCKET}/{PREFIX}a.mp4")
        self.assertEqual(entries[0].size_bytes, 569135399)
        self.assertEqual(entries[0].last_modified, datetime(2026, 6, 13, 17, 35, 52))

    def test_resolves_bare_keys_against_default_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keys.csv"
            _write_csv(path, ("key",), [{"key": f"{PREFIX}a.mp4"}])
            entries = list(ManifestReader(path, default_bucket=BUCKET))

        self.assertEqual(entries[0].source_uri, f"s3://{BUCKET}/{PREFIX}a.mp4")
        self.assertIsNone(entries[0].size_bytes)

    def test_bare_keys_stay_local_paths_without_a_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keys.csv"
            _write_csv(path, ("key",), [{"key": "fixtures/sample.mp4"}])
            entries = list(ManifestReader(path))

        self.assertEqual(entries[0].source_uri, "fixtures/sample.mp4")

    def test_derives_key_from_uri_when_no_key_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uris.csv"
            _write_csv(
                path,
                ("source_uri",),
                [{"source_uri": f"s3://{BUCKET}/{PREFIX}a.mp4"}, {"source_uri": "/tmp/local.mp4"}],
            )
            entries = list(ManifestReader(path))

        self.assertEqual(entries[0].key, f"{PREFIX}a.mp4")
        self.assertEqual(entries[1].key, "local.mp4")

    def test_reads_newline_delimited_list_skipping_comments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "videos.txt"
            path.write_text(
                f"# reference set\ns3://{BUCKET}/{PREFIX}a.mp4\n\ns3://{BUCKET}/{PREFIX}b.mp4\n",
                encoding="utf-8",
            )
            entries = list(ManifestReader(path))

        self.assertEqual([entry.key for entry in entries], [f"{PREFIX}a.mp4", f"{PREFIX}b.mp4"])

    def test_skips_blank_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            _write_csv(
                path,
                ("key", "source_uri"),
                [{"key": f"{PREFIX}a.mp4", "source_uri": ""}, {"key": "", "source_uri": ""}],
            )
            reader = ManifestReader(path, default_bucket=BUCKET)

        self.assertEqual(len(reader), 2)
        self.assertEqual(len(list(reader)), 1)

    def test_rejects_manifest_without_an_identifier_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            _write_csv(path, ("size_bytes",), [{"size_bytes": "1"}])
            with self.assertRaises(ValueError):
                ManifestReader(path)

    def test_rejects_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            ManifestReader("/nonexistent/manifest.csv")


class WriteManifestTests(unittest.TestCase):
    def test_roundtrips_entries_without_credentials(self) -> None:
        entries = [
            ManifestEntry(
                key=f"{PREFIX}a.mp4",
                source_uri=f"s3://{BUCKET}/{PREFIX}a.mp4",
                size_bytes=100,
                last_modified=datetime(2026, 6, 13, 17, 35, 52),
            ),
            ManifestEntry(key=f"{PREFIX}b.mp4", source_uri=f"s3://{BUCKET}/{PREFIX}b.mp4"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "manifest.csv"
            written = write_manifest(entries, path)
            text = path.read_text(encoding="utf-8")
            roundtripped = list(ManifestReader(path))

        self.assertEqual(written, 2)
        self.assertNotIn("X-Amz-Signature", text)
        self.assertEqual(tuple(text.splitlines()[0].split(",")), MANIFEST_FIELDS)
        self.assertEqual(
            [entry.source_uri for entry in roundtripped], [e.source_uri for e in entries]
        )
        self.assertEqual(roundtripped[0].size_bytes, 100)
        self.assertIsNone(roundtripped[1].size_bytes)


class KeyForUriTests(unittest.TestCase):
    def test_uses_s3_key_and_strips_url_query(self) -> None:
        self.assertEqual(key_for_uri(f"s3://{BUCKET}/{PREFIX}a.mp4"), f"{PREFIX}a.mp4")
        self.assertEqual(key_for_uri("https://host/path/a.mp4?X-Amz-Signature=x"), "a.mp4")
        self.assertEqual(key_for_uri("/tmp/a.mp4"), "a.mp4")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
