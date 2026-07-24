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

    The final window is anchored to the end of the video.  This retains the
    desired uniform samples while ensuring that an event at the end of a video
    is not systematically excluded by a stride larger than the window length.
    """

    def __init__(self, window_duration: float, stride: float) -> None:
        if not isfinite(window_duration) or window_duration <= 0:
            raise ValueError("window_duration must be a finite value greater than zero")
        if not isfinite(stride) or stride <= 0:
            raise ValueError("stride must be a finite value greater than zero")

        self.window_duration = window_duration
        self.stride = stride

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
