"""Reproducible schedules for selecting temporal video windows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import isclose, isfinite


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """A time range to be decoded later by :class:`VideoReader`."""

    start_time: float
    end_time: float


class WindowSampler:
    """Create deterministic, uniformly spaced window schedules.

    ``stride`` of ``None`` means contiguous: the stride equals the window, so the
    schedule tiles the video end to end and nothing between the first and last
    window goes unmeasured.  That is the default, and it is expressed as ``None``
    rather than as a number that happens to match ``window_duration`` so the two
    cannot drift apart -- changing the window length alone used to silently
    reintroduce gaps or overlap.

    A stride larger than the window samples the video instead of covering it,
    which is faster but can miss an artifact entirely between two windows.

    The final window is anchored to the end of the video.  Under a sampling
    stride that stops an event in the last seconds from being systematically
    excluded; under a contiguous stride it covers the remainder that does not
    divide evenly into whole windows, overlapping its predecessor by less than
    one window.  Either way every window is exactly ``window_duration`` long,
    which is what keeps their scores comparable now that each one is reported
    and graded in its own right.
    """

    def __init__(self, window_duration: float, stride: float | None = None) -> None:
        if not isfinite(window_duration) or window_duration <= 0:
            raise ValueError("window_duration must be a finite value greater than zero")
        if stride is None:
            stride = window_duration
        elif not isfinite(stride) or stride <= 0:
            raise ValueError("stride must be a finite value greater than zero, or null")

        self.window_duration = window_duration
        self.stride = stride

    @property
    def is_contiguous(self) -> bool:
        """Whether the schedule covers the video rather than sampling it."""
        return isclose(self.stride, self.window_duration, rel_tol=0.0, abs_tol=1e-9)

    def sample(self, video_duration: float) -> Iterator[WindowSpec]:
        """Yield windows for ``video_duration`` in deterministic time order.

        A video shorter than ``window_duration`` yields its one available,
        shorter window.  Otherwise all windows have exactly
        ``window_duration`` seconds, including the final end-anchored window.
        """
        if not isfinite(video_duration) or video_duration <= 0:
            raise ValueError("video_duration must be a finite value greater than zero")

        last_start = max(video_duration - self.window_duration, 0.0)
        current = 0.0

        while current < last_start and not isclose(
            current,
            last_start,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            yield WindowSpec(
                start_time=current,
                end_time=current + self.window_duration,
            )
            current += self.stride

        yield WindowSpec(start_time=last_start, end_time=video_duration)
