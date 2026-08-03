#!/usr/bin/env python3
"""Extract the reference label set from the delivery spreadsheet's cell colours.

The labels exist only as fill colours, which no CSV export preserves, so this is
a one-off data-prep step whose output is committed alongside the code.

Usage:
    uv run python scripts/extract_labels.py \
        data/visionlabs_500h_delivery.xlsx \
        --output data/reference_labels.csv
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.reference_sheet import extract_labels, write_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", help="Colour-coded delivery spreadsheet (.xlsx)")
    parser.add_argument("--output", default="data/reference_labels.csv")
    arguments = parser.parse_args()

    labels = extract_labels(arguments.workbook)
    if not labels:
        raise SystemExit(
            f"No colour-coded rows found in {arguments.workbook}. "
            "Confirm the sheet still uses solid red/orange/green fills."
        )
    written = write_labels(labels, arguments.output)
    counts = collections.Counter(label for _, label in labels)
    print(f"Wrote {written} labels to {arguments.output}")
    for label in ("none", "mild", "extreme"):
        print(f"  {label:8} {counts.get(label, 0)}")


if __name__ == "__main__":
    main()
