from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.core.types import ManifestEntry


class ManifestReader:
    """
    Reads the dataset manifest and yields validated ManifestEntry objects.

    Expected columns:
        - key
        - size_bytes
        - last_modified
        - presigned_url_7day
    """

    REQUIRED_COLUMNS = {
        "key",
        "size_bytes",
        "last_modified",
        "presigned_url_7day",
    }

    def __init__(self, manifest_path: str | Path):

        self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)

        missing = self.REQUIRED_COLUMNS.difference(self.df.columns)

        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[ManifestEntry]:

        for row in self.df.itertuples(index=False):
            yield ManifestEntry(
                key=row.key,
                size_bytes=int(row.size_bytes),
                last_modified=datetime.fromisoformat(str(row.last_modified)),
                presigned_url=row.presigned_url_7day,
            )
