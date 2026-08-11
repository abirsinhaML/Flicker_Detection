"""Read the work set from a link sheet instead of enumerating bucket prefixes.

The corpus is named by a spreadsheet whose ``video_s3_link`` column holds one
bucket-relative key per video.  That is the input: the key comes from the column
and the bucket from ``s3.bucket``, so the work set is a fixed, reviewable list
rather than whatever a prefix happens to contain at listing time.  Prefix
enumeration remains available for discovering objects the sheet does not name.

One reconciliation stands between the sheet and a working URI.  The sheet's links
are lowercased and S3 keys are case-sensitive, so joining bucket and column
verbatim 404s on every row -- measured against the live bucket, all of them.  The
real keys carry mixed case (``raw/Delhi_ZetWork/2026-06-26/EL0706/GX010086.MP4``
against the sheet's ``raw/delhi_zetwork/2026-06-26/el0706/gx010086.mp4``), so the
casing has to be recovered from the bucket itself.  One listing pass builds a
case-folded index, and 140,335 of 140,519 rows (99.87%) then resolve to exactly
one real object.

The residue is reported, never guessed at:

* **missing** -- no object at any casing.  The sheet names a video the bucket
  does not hold, so there is nothing to score.
* **ambiguous** -- two real objects differ only in case, as
  ``mirana_GoPro_IMU_SD21_GX040004`` and ``mirana_GoPro_IMU_sd21_GX040004`` do.
  A lowercased link cannot pick between them and this module will not pick for
  it; scoring an arbitrary one would attach a verdict to the wrong video.

Set ``resolve_case=False`` once the sheet carries exact keys, and the listing
pass disappears.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.core.types import ManifestEntry
from src.data.rows import describe_row_range, row_slice
from src.data.s3_source import S3VideoCatalog, build_s3_uri, is_s3_uri

logger = logging.getLogger(__name__)

_SHEET_SUFFIXES = frozenset({".xlsx", ".xlsm", ".xls"})

# ``video_s3_link`` is this corpus's column; the rest are accepted so a sheet
# exported under another name still reads without being renamed first.
LINK_COLUMNS: tuple[str, ...] = (
    "video_s3_link",
    "video_link",
    "s3_link",
    "s3_uri",
    "source_uri",
)
PROJECT_COLUMNS: tuple[str, ...] = ("project_name", "project")
VIDEO_ID_COLUMNS: tuple[str, ...] = ("video_id", "id")
DURATION_COLUMNS: tuple[str, ...] = ("duration_s", "duration_seconds", "duration")


@dataclass(slots=True)
class LinkResolution:
    """What became of every row of the sheet.

    Kept as lists rather than counts so the unusable rows can be written out and
    chased, instead of vanishing into a log line.
    """

    total: int = 0
    resolved: int = 0
    unchanged: int = 0
    recased: int = 0
    missing: list[str] = field(default_factory=list)
    ambiguous: list[tuple[str, list[str]]] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> int:
        return len(self.missing) + len(self.ambiguous) + len(self.duplicates)

    def summary(self) -> str:
        return (
            f"{self.resolved} of {self.total} links resolved "
            f"({self.recased} re-cased, {self.unchanged} already exact); "
            f"skipped {self.skipped}: {len(self.missing)} missing, "
            f"{len(self.ambiguous)} ambiguous, {len(self.duplicates)} duplicate"
        )


@dataclass(frozen=True, slots=True)
class LinkRecord:
    """One row of the sheet: a key, plus whatever context it carries."""

    link: str
    project_name: str | None = None
    video_id: str | None = None
    duration_seconds: float | None = None


class LinkSheetReader:
    """Read ``video_s3_link`` and its sibling columns from a sheet or CSV.

    Rows are yielded in file order and streamed off the frame, so the sheet is
    parsed once and never copied into a second full-size structure.

    ``from_row`` and ``to_row`` select an inclusive, zero-based range of the
    sheet's rows, which is how a run is sharded across instances.
    """

    def __init__(
        self,
        sheet_path: str | Path,
        *,
        from_row: int | None = None,
        to_row: int | None = None,
    ) -> None:
        self.sheet_path = Path(sheet_path)
        if not self.sheet_path.exists():
            raise FileNotFoundError(f"Link sheet not found: {self.sheet_path}")

        frame = self._read(self.sheet_path)
        # Sliced here, on the sheet's own rows, so a row number means the same
        # thing to every instance regardless of what resolves later.
        self.total_rows = len(frame)
        self.frame = row_slice(frame, from_row, to_row)
        self.row_range = describe_row_range(self.total_rows, len(self.frame), from_row, to_row)
        self.link_column = _first_present(self.frame, LINK_COLUMNS)
        if self.link_column is None:
            raise ValueError(
                f"Link sheet {self.sheet_path} needs one of {list(LINK_COLUMNS)}; "
                f"found {sorted(self.frame.columns)}"
            )
        self.project_column = _first_present(self.frame, PROJECT_COLUMNS)
        self.video_id_column = _first_present(self.frame, VIDEO_ID_COLUMNS)
        self.duration_column = _first_present(self.frame, DURATION_COLUMNS)

    def __len__(self) -> int:
        return len(self.frame)

    def __iter__(self) -> Iterator[LinkRecord]:
        # Only the columns in use are walked, and as tuples rather than dicts, so
        # a 140K-row sheet is not copied into a second full-size structure.
        wanted = {
            "link": self.link_column,
            "project": self.project_column,
            "video_id": self.video_id_column,
            "duration": self.duration_column,
        }
        present = {name: column for name, column in wanted.items() if column is not None}
        for row in self.frame[list(present.values())].itertuples(index=False, name=None):
            values = dict(zip(present, row, strict=True))
            link = _clean(values.get("link"))
            if not link:
                continue
            yield LinkRecord(
                link=link,
                project_name=_clean(values.get("project")),
                video_id=_clean(values.get("video_id")),
                duration_seconds=_optional_float(values.get("duration")),
            )

    @staticmethod
    def _read(sheet_path: Path) -> pd.DataFrame:
        if sheet_path.suffix.lower() in _SHEET_SUFFIXES:
            frame = pd.read_excel(sheet_path)
        else:
            frame = pd.read_csv(sheet_path)
        if not len(frame.columns):
            raise ValueError(f"Link sheet is empty: {sheet_path}")
        return frame


class BucketKeyIndex:
    """Case-folded index of a bucket's real object keys.

    Built by listing once.  The index is what recovers the casing the sheet lost,
    and it deliberately keeps *every* real key a folded key maps to, so a genuine
    collision is reported rather than silently resolved to whichever object the
    listing happened to reach first.
    """

    def __init__(self, keys: Iterable[str]) -> None:
        self._index: dict[str, list[str]] = {}
        for key in keys:
            self._index.setdefault(key.lower(), []).append(key)

    def __len__(self) -> int:
        return len(self._index)

    @classmethod
    def from_catalog(cls, catalog: S3VideoCatalog) -> BucketKeyIndex:
        """List the catalog's prefix and index every video key under it."""
        logger.info(
            "Indexing object keys under s3://%s/%s to recover link casing",
            catalog.bucket,
            catalog.prefix,
        )
        index = cls(entry.key for entry in catalog.list_entries())
        logger.info("Indexed %d distinct object keys", len(index))
        return index

    def lookup(self, key: str) -> list[str]:
        """Return every real key matching ``key`` ignoring case."""
        exact = self._index.get(key.lower())
        if not exact:
            return []
        # An exact hit wins outright: if the sheet already names the real key,
        # a case collision elsewhere in the bucket is not this row's problem.
        return [key] if key in exact else list(exact)


def resolve_links(
    records: Iterable[LinkRecord],
    bucket: str,
    *,
    index: BucketKeyIndex | None = None,
    resolution: LinkResolution | None = None,
) -> Iterator[ManifestEntry]:
    """Turn sheet rows into entries addressing real objects in ``bucket``.

    Yields lazily, so a 140K-row sheet costs no more memory than the frame it was
    read from and work starts before the whole sheet has been walked.  Rows that
    cannot be resolved are recorded in ``resolution`` and skipped: an unresolvable
    link is not a video, and scoring it would only write a 404 into the output
    that ``--resume`` then retries forever.

    ``index`` omitted means the links are taken to be exact, which is correct
    once the sheet carries real keys and skips the listing pass entirely.
    """
    report = resolution if resolution is not None else LinkResolution()
    seen: set[str] = set()

    for record in records:
        report.total += 1
        link = record.link
        # A full URI in the column addresses its own object and needs no bucket.
        if is_s3_uri(link) or "://" in link:
            source_uri, key = link, link
        else:
            key = link.lstrip("/")
            if index is not None:
                matches = index.lookup(key)
                if not matches:
                    report.missing.append(link)
                    continue
                if len(matches) > 1:
                    report.ambiguous.append((link, matches))
                    continue
                if matches[0] != key:
                    report.recased += 1
                else:
                    report.unchanged += 1
                key = matches[0]
            else:
                report.unchanged += 1
            source_uri = build_s3_uri(bucket, key)

        if key in seen:
            report.duplicates.append(link)
            continue
        seen.add(key)
        report.resolved += 1
        yield ManifestEntry(
            key=key,
            source_uri=source_uri,
            project_name=record.project_name,
            video_id=record.video_id,
            duration_seconds=record.duration_seconds,
        )


def write_resolution_report(
    resolution: LinkResolution,
    output_path: str | Path,
    *,
    sheet_path: str | Path | None = None,
    limit: int = 200,
) -> None:
    """Write every unresolvable link out, so the gap can be chased.

    Truncated at ``limit`` per category with the true count stated, because a
    report nobody can open is not a report -- and a silent truncation would read
    as "that was all of them".
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Link resolution",
        "",
        f"Sheet: `{sheet_path}`" if sheet_path else "",
        "",
        f"- rows: {resolution.total}",
        f"- resolved: {resolution.resolved} "
        f"({resolution.recased} re-cased, {resolution.unchanged} already exact)",
        f"- skipped: {resolution.skipped}",
        "",
    ]
    sections = (
        (
            "Missing",
            "No object at any casing, so there is nothing to score.",
            resolution.missing,
        ),
        (
            "Ambiguous",
            "Two or more real objects differ only in case; a lowercased link "
            "cannot pick between them, and picking arbitrarily would attach a "
            "verdict to the wrong video.",
            [f"{link} -> {', '.join(matches)}" for link, matches in resolution.ambiguous],
        ),
        (
            "Duplicate",
            "The same object is named more than once; only the first is scored.",
            resolution.duplicates,
        ),
    )
    for title, explanation, items in sections:
        lines += [f"## {title} ({len(items)})", "", explanation, ""]
        for item in items[:limit]:
            lines.append(f"- `{item}`")
        if len(items) > limit:
            lines.append(f"- ...and {len(items) - limit} more")
        lines.append("")

    output_path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    logger.info("Wrote link resolution report: %s", output_path)


def _first_present(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in lowered:
            return str(lowered[candidate])
    return None


def _clean(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if parsed == parsed else None
