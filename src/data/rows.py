"""Selecting a contiguous row range of an input file, for sharding a run.

One instance cannot decode a 28,000-hour corpus in reasonable time, and the work
divides cleanly: give each instance a row range of the link sheet and they cover
the corpus between them with no coordination.

Row numbers address *file rows*, counted from zero, and the range is inclusive at
both ends -- ``0..99`` then ``100..199`` tile without overlap.  The slice is taken
on the input file before anything is resolved or filtered, which is what makes the
ranges stable: rows that turn out to be unresolvable simply leave their instance
with less to do rather than shifting every later boundary and letting two
instances collide.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from itertools import accumulate

import pandas as pd

logger = logging.getLogger(__name__)


def validate_row_range(from_row: int | None, to_row: int | None) -> None:
    """Reject a range that cannot select anything, before any work starts."""
    if from_row is not None and from_row < 0:
        raise ValueError("--from-row must be zero or greater")
    if to_row is not None and to_row < 0:
        raise ValueError("--to-row must be zero or greater")
    if from_row is not None and to_row is not None and to_row < from_row:
        raise ValueError(f"--to-row ({to_row}) must not be below --from-row ({from_row})")


def row_slice(
    frame: pd.DataFrame,
    from_row: int | None = None,
    to_row: int | None = None,
) -> pd.DataFrame:
    """Return the inclusive ``[from_row, to_row]`` rows of ``frame``.

    Either bound may be omitted to run from the start or to the end.  A range
    beyond the last row yields nothing rather than raising, so the final shard of
    a fleet can be given a generous upper bound.
    """
    validate_row_range(from_row, to_row)
    if from_row is None and to_row is None:
        return frame
    start = from_row or 0
    # +1 because the upper bound is inclusive, unlike a Python slice.
    stop = None if to_row is None else to_row + 1
    return frame.iloc[start:stop]


def describe_row_range(
    total: int,
    selected: int,
    from_row: int | None,
    to_row: int | None,
) -> str:
    """One line naming exactly which rows were taken, for the run's log.

    Worth logging explicitly: an off-by-one in a fleet of ten instances leaves a
    silent gap in the corpus or scores the same videos twice, and neither shows up
    anywhere else.
    """
    if from_row is None and to_row is None:
        return f"all {total} rows"
    first = from_row or 0
    last = "end" if to_row is None else to_row
    return f"rows {first}..{last} inclusive: {selected} of {total}"


def weighted_shard_ranges(weights: Sequence[float], shards: int) -> list[tuple[int, int]]:
    """Split rows into ``shards`` contiguous ranges of near-equal total weight.

    Equal *row counts* do not mean equal work: this corpus runs from 0.02 s to
    3.7 hours per video, and cost is proportional to duration because decode is
    ~94% of it.  Splitting 140,519 rows six ways by count leaves the heaviest
    shard with 1.20x the hours of the lightest, so a fleet spends a fifth of its
    wall time with an instance already finished and idle.  Weighting by duration
    removes that.

    Ranges stay contiguous, so they are still expressible as
    ``--from-row``/``--to-row`` and still tile the input exactly.

    At each boundary the split takes whichever side of the target is *closer*
    rather than the first row past it.  Always closing after the crossing
    overshoots systematically -- on ``[1]*5 + [100]*5`` split in two it gives
    305/200 where 205/300 was available -- and the bias compounds across shards,
    loading the early ones.
    """
    if shards < 1:
        raise ValueError("shards must be at least one")
    total_rows = len(weights)
    if total_rows == 0:
        return []
    if shards >= total_rows:
        return [(index, index) for index in range(total_rows)]

    # Negative weights would break the monotonicity the scan below relies on;
    # a duration cannot be negative, so clamping is the honest reading of one.
    values = [max(0.0, float(weight)) for weight in weights]
    total_weight = sum(values)
    if total_weight <= 0:
        return shard_ranges(total_rows, shards)

    prefix = list(accumulate(values))
    ranges: list[tuple[int, int]] = []
    start = 0
    for shard in range(shards - 1):
        target = total_weight * (shard + 1) / shards
        # The last row this shard may take, leaving one for each shard after it.
        highest = total_rows - shards + shard
        best, best_distance = start, abs(prefix[start] - target)
        for index in range(start, highest + 1):
            distance = abs(prefix[index] - target)
            if distance < best_distance:
                best, best_distance = index, distance
            # prefix only grows, so once past the target nothing later is closer.
            if prefix[index] >= target:
                break
        ranges.append((start, best))
        start = best + 1

    ranges.append((start, total_rows - 1))
    return ranges


def shard_ranges(total: int, shards: int) -> list[tuple[int, int]]:
    """Split ``total`` rows into ``shards`` inclusive ranges that tile exactly.

    Computed rather than typed by hand, because hand-written ranges across a
    fleet are where gaps and overlaps come from.  Earlier shards take the
    remainder, so no shard is empty while another has two rows.
    """
    if shards < 1:
        raise ValueError("shards must be at least one")
    if total < 0:
        raise ValueError("total must not be negative")
    base, remainder = divmod(total, shards)
    ranges: list[tuple[int, int]] = []
    start = 0
    for index in range(shards):
        size = base + (1 if index < remainder else 0)
        if size == 0:
            continue
        ranges.append((start, start + size - 1))
        start += size
    return ranges
