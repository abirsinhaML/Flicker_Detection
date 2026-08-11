"""Durable output for a batch: newline-delimited records plus flat exports.

The primary artifact is JSON Lines, one line per video, each line carrying that
video's rollup and every one of its windows.  That shape is chosen for the
property the batch depends on: a video is written in exactly one atomic append,
so a batch killed mid-run leaves no half-written video and ``--resume`` keeps its
video-level granularity even though the pipeline now emits ~200 rows per video.

One shared per-window CSV would give up that property -- a video's rows would
interleave with other workers' and a kill could truncate a video mid-way through
its windows, leaving rows that look complete.  The flat tables are therefore
*derived*, written alongside the JSONL or exported from it afterwards.

Window metrics are written one CSV per video, at the video's own key path under a
root directory, so a result is addressed exactly like the video it describes and
no two workers ever write the same file.  Concatenating the tree reproduces the
single combined table, which :func:`write_window_csv` still exports for ad-hoc
analysis.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, TextIO

from src.core.records import (
    VIDEO_FIELDS,
    WINDOW_FIELDS,
    video_row,
    window_csv_path,
    window_rows,
)
from src.data.publish import S3Publisher

logger = logging.getLogger(__name__)


def dump_record(record: Mapping[str, Any]) -> str:
    """Serialize one record to a single line of JSON."""
    return json.dumps(record, separators=(",", ":"), allow_nan=False)


def append_record(output_file: TextIO, record: Mapping[str, Any]) -> None:
    """Append one record and flush, so a killed batch loses nothing complete."""
    output_file.write(dump_record(record) + "\n")
    output_file.flush()


def read_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield every parseable record from a JSONL file.

    A truncated or corrupt final line is skipped with a warning rather than
    failing the read: the file is written by a long batch that may have been
    killed, and one unusable line must not cost the other quarter-million.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(encoding="utf-8") as records_file:
        for number, line in enumerate(records_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping unparseable record at %s:%d", path, number)
                continue
            if isinstance(record, dict):
                yield record


def read_successful_records(path: str | Path) -> dict[str, dict[str, Any]]:
    """Return the successful records of an existing output, keyed by video.

    Only ``status == "ok"`` counts as done.  The expected failures here are
    transient -- expired credentials, throttling, a dropped connection part-way
    through an object -- so they are retried.  Duplicate keys collapse to the
    last record, which is how a re-run supersedes an earlier one.
    """
    records: dict[str, dict[str, Any]] = {}
    for record in read_records(path):
        key = str(record.get("video_key") or "").strip()
        if key and record.get("status") == "ok":
            records[key] = record
    return records


class ResultWriter:
    """Write records as JSONL, and derive the flat tables as it goes.

    The JSONL file is opened for the batch's lifetime and appended per video, as
    is the per-video rollup CSV, which is rewritten from scratch on open with any
    retained records so exactly one row per key survives a resume.

    Window metrics go to one CSV per video under ``window_dir``, named for the
    video's key.  Each worker's video owns its own file, so nothing interleaves
    and a killed batch leaves complete files rather than a truncated table.
    """

    def __init__(
        self,
        records_path: str | Path,
        *,
        video_csv_path: str | Path | None = None,
        window_dir: str | Path | None = None,
        publisher: S3Publisher | None = None,
        publish_root: str | Path | None = None,
    ) -> None:
        self.records_path = Path(records_path)
        self.video_csv_path = Path(video_csv_path) if video_csv_path else None
        self.window_dir = Path(window_dir) if window_dir else None
        # A per-video CSV is final the moment it is written, so it is copied up
        # immediately and a long batch accumulates results remotely as it runs.
        # The JSONL and the rollup CSV grow all run and are copied once at the
        # end by publish_summary; re-uploading a 25 GB JSONL per video would cost
        # more than the batch.
        self.publisher = publisher
        self.publish_root = Path(publish_root) if publish_root else None
        self._records_file: TextIO | None = None
        self._video_file: TextIO | None = None
        self._video_writer: csv.DictWriter | None = None

    def open(self, retained: Iterable[Mapping[str, Any]] = ()) -> ResultWriter:
        """Truncate the outputs and re-emit ``retained`` before new work lands."""
        for path in (self.records_path, self.video_csv_path):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        if self.window_dir is not None:
            self.window_dir.mkdir(parents=True, exist_ok=True)

        self._records_file = self.records_path.open("w", encoding="utf-8")
        if self.video_csv_path is not None:
            self._video_file = self.video_csv_path.open("w", newline="", encoding="utf-8")
            self._video_writer = csv.DictWriter(self._video_file, fieldnames=VIDEO_FIELDS)
            self._video_writer.writeheader()

        for record in retained:
            # A resumed video's own CSV is normally already on disk, and a batch
            # resuming 140K of them must not rewrite 140K files to learn that.
            # Only a missing one is regenerated, which keeps the tree complete
            # without making resume proportional to work already done.
            self.write(record, rewrite_window_csv=False)
        return self

    def write(self, record: Mapping[str, Any], *, rewrite_window_csv: bool = True) -> None:
        """Write one video's record to every configured output."""
        if self._records_file is None:
            raise RuntimeError("ResultWriter.open must be called before write")
        append_record(self._records_file, record)
        if self._video_writer is not None and self._video_file is not None:
            self._video_writer.writerow(video_row(record))
            self._video_file.flush()
        if self.window_dir is not None:
            self._write_window_csv(record, rewrite=rewrite_window_csv)

    def _write_window_csv(self, record: Mapping[str, Any], *, rewrite: bool) -> None:
        """Write one video's windows to their own CSV, mirroring the key path.

        A failed video yields no rows and therefore no file: an empty CSV would
        claim a video was measured and found clean.
        """
        if self.window_dir is None or record.get("status") != "ok":
            return
        key = str(record.get("video_key") or "").strip()
        if not key:
            logger.warning("Record without a video_key; skipping its window CSV")
            return
        path = window_csv_path(key, self.window_dir)
        if not rewrite and path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as window_file:
            writer = csv.DictWriter(window_file, fieldnames=WINDOW_FIELDS)
            writer.writeheader()
            writer.writerows(window_rows(record))
        self._publish(path)

    def _publish(self, path: Path) -> None:
        """Copy one written file to S3, if a destination was configured."""
        if self.publisher is None or self.publish_root is None:
            return
        try:
            relative = path.resolve().relative_to(self.publish_root.resolve())
        except ValueError:
            logger.warning("Not publishing %s: outside the output root", path)
            return
        self.publisher.upload(path, relative)

    def publish_summary(self) -> None:
        """Copy the run-level files up, once, after the batch has finished.

        Separate from :meth:`write` because these files are appended to all run:
        uploading them per video would repeat the whole file every time.
        """
        if self.publisher is None:
            return
        for path in (self.records_path, self.video_csv_path):
            if path is not None and path.is_file():
                self._publish(path)

    def close(self) -> None:
        for handle in (self._records_file, self._video_file):
            if handle is not None:
                handle.close()
        self._records_file = self._video_file = None
        self._video_writer = None

    def __enter__(self) -> ResultWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_video_csv(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Export the one-row-per-video rollup table."""
    return _write_csv(path, VIDEO_FIELDS, (video_row(record) for record in records))


def write_window_csv(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Export every window into one combined analytics table."""
    return _write_csv(
        path,
        WINDOW_FIELDS,
        (row for record in records for row in window_rows(record)),
    )


def write_window_tree(root: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Export one window CSV per video, mirroring each key's path under ``root``.

    Regenerates from the JSONL what a batch writes as it runs, so a tree that was
    never written, was pruned, or predates a schema change can be rebuilt without
    re-decoding anything.  Returns the number of files written.
    """
    written = 0
    for record in records:
        if record.get("status") != "ok":
            continue
        key = str(record.get("video_key") or "").strip()
        if not key:
            continue
        _write_csv(window_csv_path(key, root), WINDOW_FIELDS, window_rows(record))
        written += 1
    return written


def _write_csv(
    path: str | Path,
    fieldnames: tuple[str, ...],
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            written += 1
    return written
