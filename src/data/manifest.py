"""Read and write manifests of durable video identifiers.

A manifest names videos; it never carries credentials.  Presigned URLs expire
within days, so the schema stores ``s3://bucket/key`` URIs and signing is
deferred to the worker that decodes the video.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.types import ManifestEntry
from src.data.rows import describe_row_range, row_slice
from src.data.s3_source import build_s3_uri, is_s3_uri, parse_s3_uri

MANIFEST_FIELDS: tuple[str, ...] = (
    "key",
    "source_uri",
    "size_bytes",
    "last_modified",
    # Carried so a manifest pinned from the link sheet is equivalent to reading
    # the sheet: without these, snapshotting would silently drop the project
    # grouping and the sheet's stated duration.
    "project_name",
    "video_id",
    "duration_s",
)
_TEXT_SUFFIXES = frozenset({".txt", ".list", ".lst"})


class ManifestReader:
    """Read a manifest of videos into validated :class:`ManifestEntry` records.

    Accepted inputs:

    * CSV with a ``source_uri`` column (preferred; ``key`` optional), or
    * CSV with only a ``key`` column, resolved against ``default_bucket``, or
    * a newline-delimited ``.txt`` list of URIs, S3 keys, or local paths.

    Each identifier may be an ``s3://bucket/key`` URI, a bare S3 key, a local
    path, or any URL FFmpeg accepts.  Optional ``size_bytes`` and
    ``last_modified`` columns are carried through when present.
    """

    URI_COLUMNS: tuple[str, ...] = ("source_uri", "s3_uri", "uri", "url")
    KEY_COLUMNS: tuple[str, ...] = ("key", "video_key")

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        default_bucket: str | None = None,
        from_row: int | None = None,
        to_row: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.default_bucket = default_bucket
        frame = self._read(self.manifest_path)
        # Same inclusive row semantics as the link sheet, so a pinned manifest
        # shards across instances exactly the way the sheet does.
        self.total_rows = len(frame)
        self.df = row_slice(frame, from_row, to_row)
        self.row_range = describe_row_range(self.total_rows, len(self.df), from_row, to_row)
        self.uri_column = self._first_present(self.URI_COLUMNS)
        self.key_column = self._first_present(self.KEY_COLUMNS)
        if self.uri_column is None and self.key_column is None:
            raise ValueError(
                f"Manifest {self.manifest_path} needs one of "
                f"{sorted({*self.URI_COLUMNS, *self.KEY_COLUMNS})}; "
                f"found {sorted(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[ManifestEntry]:
        for row in self.df.to_dict(orient="records"):
            entry = self._to_entry(row)
            if entry is not None:
                yield entry

    def _to_entry(self, row: dict[str, object]) -> ManifestEntry | None:
        identifier = _clean(row.get(self.uri_column) if self.uri_column else None)
        key = _clean(row.get(self.key_column) if self.key_column else None)
        if not identifier and not key:
            return None

        source_uri = identifier or self._source_uri_for_key(str(key))
        return ManifestEntry(
            key=key or key_for_uri(source_uri),
            source_uri=source_uri,
            size_bytes=_optional_int(row.get("size_bytes")),
            last_modified=_optional_datetime(row.get("last_modified")),
            project_name=_clean(row.get("project_name")),
            video_id=_clean(row.get("video_id")),
            duration_seconds=_optional_float(row.get("duration_s")),
        )

    def _source_uri_for_key(self, key: str) -> str:
        """Turn a bare key into a source URI, preferring the default bucket."""
        if is_s3_uri(key) or "://" in key:
            return key
        if self.default_bucket:
            return build_s3_uri(self.default_bucket, key)
        # No bucket to resolve against: treat the value as a local path, which
        # is how the test fixtures and local reruns address videos.
        return key

    def _first_present(self, candidates: Sequence[str]) -> str | None:
        return next((column for column in candidates if column in self.df.columns), None)

    @staticmethod
    def _read(manifest_path: Path) -> pd.DataFrame:
        if manifest_path.suffix.lower() in _TEXT_SUFFIXES:
            lines = [
                line.strip()
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            return pd.DataFrame({"source_uri": lines})

        frame = pd.read_csv(manifest_path)
        if frame.empty and not len(frame.columns):
            raise ValueError(f"Manifest is empty: {manifest_path}")
        return frame


def write_manifest(entries: Iterable[ManifestEntry], output_path: str | Path) -> int:
    """Write entries as a durable, credential-free CSV manifest.

    Returns the number of rows written.  Streams row by row so that snapshotting
    a very large prefix does not require holding the listing in memory.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    "key": entry.key,
                    "source_uri": entry.source_uri,
                    "size_bytes": "" if entry.size_bytes is None else entry.size_bytes,
                    "last_modified": (
                        "" if entry.last_modified is None else entry.last_modified.isoformat()
                    ),
                    "project_name": entry.project_name or "",
                    "video_id": entry.video_id or "",
                    "duration_s": (
                        "" if entry.duration_seconds is None else entry.duration_seconds
                    ),
                }
            )
            written += 1
    return written


def key_for_uri(source_uri: str) -> str:
    """Derive a stable, human-meaningful key from a source URI."""
    if is_s3_uri(source_uri):
        return parse_s3_uri(source_uri).key
    return Path(source_uri.split("?", 1)[0]).name or source_uri


def _clean(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _optional_float(value: object) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _optional_datetime(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = _clean(value)
    if text is None:
        return None
    try:
        parsed = pd.to_datetime(text)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(parsed) else parsed.to_pydatetime()
