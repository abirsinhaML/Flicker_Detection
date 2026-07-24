"""Video decoding behind a small, backend-independent reader interface."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import av
import numpy as np

from src.core.types import VideoMetadata, VideoWindow


class VideoReader:
    """Read metadata and time-bounded RGB frame windows from one video source.

    ``video_path`` may be a local path or any URL accepted by FFmpeg, including a
    presigned object-storage URL.  The reader owns an open PyAV container and
    should therefore be used as a context manager where possible.
    """

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = str(video_path)
        self.container = av.open(self.video_path)

        if not self.container.streams.video:
            self.container.close()
            raise ValueError(f"No video stream found in: {self.video_path}")

        self.stream = self.container.streams.video[0]

    def metadata(self) -> VideoMetadata:
        """Return metadata reported by the selected video stream."""
        if self.stream.average_rate is None:
            raise ValueError(f"Video stream has no average frame rate: {self.video_path}")
        if self.stream.duration is None:
            raise ValueError(f"Video stream has no duration: {self.video_path}")

        return VideoMetadata(
            fps=float(self.stream.average_rate),
            frame_count=self.stream.frames,
            duration=float(self.stream.duration * self.stream.time_base),
            width=self.stream.width,
            height=self.stream.height,
        )

    def read_window(
        self,
        start: float,
        duration: float,
        resize: tuple[int, int] | None = None,
    ) -> VideoWindow:
        """Decode RGB frames in the half-open interval ``[start, start + duration)``.

        Seeking starts from a decodable key frame; decoded frames before ``start``
        are discarded so the returned window honors the requested timestamp even
        when the source has sparse key frames or a variable frame rate.  ``resize``
        is ``(width, height)`` and is performed by FFmpeg/PyAV.
        """
        self._validate_window(start, duration, resize)

        end = start + duration
        self.container.seek(
            int(start / self.stream.time_base),
            stream=self.stream,
            backward=True,
            any_frame=False,
        )

        frames: list[np.ndarray] = []
        for frame in self.container.decode(self.stream):
            timestamp = frame.time
            if timestamp is None:
                continue

            if timestamp < start:
                continue
            if timestamp >= end:
                break

            if resize is not None:
                frame = frame.reformat(
                    width=resize[0],
                    height=resize[1],
                    format="rgb24",
                )

            frames.append(frame.to_ndarray(format="rgb24"))

        frame_width, frame_height = resize or (self.stream.width, self.stream.height)
        rgb_frames = (
            np.stack(frames)
            if frames
            else np.empty((0, frame_height, frame_width, 3), dtype=np.uint8)
        )

        return VideoWindow(
            start_time=start,
            end_time=end,
            fps=self._fps(),
            frames=rgb_frames,
        )

    def close(self) -> None:
        """Release the underlying media container."""
        self.container.close()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _fps(self) -> float:
        if self.stream.average_rate is None:
            raise ValueError(f"Video stream has no average frame rate: {self.video_path}")
        return float(self.stream.average_rate)

    @staticmethod
    def _validate_window(
        start: float,
        duration: float,
        resize: tuple[int, int] | None,
    ) -> None:
        if not isfinite(start) or start < 0:
            raise ValueError("start must be a finite, non-negative value")
        if not isfinite(duration) or duration <= 0:
            raise ValueError("duration must be a finite value greater than zero")
        if resize is not None and (len(resize) != 2 or min(resize) <= 0):
            raise ValueError("resize must contain positive (width, height) values")
