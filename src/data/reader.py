"""Video decoding behind a small, backend-independent reader interface."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import av
import numpy as np

from src.core.types import VideoMetadata, VideoWindow

_REMOTE_SCHEMES = ("http://", "https://")

# Windowed sampling seeks repeatedly, which over HTTP means a byte-range request
# per window.  ``multiple_requests`` keeps one connection alive across those
# seeks instead of reopening the object each time, and the reconnect options
# absorb the transient resets normal on long-lived object-storage reads.
_HTTP_OPTIONS = {
    "multiple_requests": "1",
    "reconnect": "1",
    "reconnect_streamed": "1",
    "reconnect_on_network_error": "1",
    "reconnect_delay_max": "8",
}
# (open, read) seconds.  Bounded so one stalled object cannot pin a worker.
_REMOTE_TIMEOUT = (15.0, 60.0)


class VideoReader:
    """Read metadata and time-bounded RGB frame windows from one video source.

    ``video_path`` may be a local path or any URL accepted by FFmpeg, including
    the short-lived signed URL of an S3 object.  Remote sources are opened with
    range-request-friendly options so that windowed sampling transfers only the
    bytes it analyzes rather than the whole object.  The reader owns an open
    PyAV container and should therefore be used as a context manager.
    """

    def __init__(self, video_path: str | Path) -> None:
        self.video_path = str(video_path)
        self.container = (
            av.open(self.video_path, options=dict(_HTTP_OPTIONS), timeout=_REMOTE_TIMEOUT)
            if self.is_remote(self.video_path)
            else av.open(self.video_path)
        )

        if not self.container.streams.video:
            self.container.close()
            raise ValueError(f"No video stream found in: {self.video_path}")

        self.stream = self.container.streams.video[0]

    @staticmethod
    def is_remote(video_path: str) -> bool:
        """Return whether FFmpeg will read this source over HTTP."""
        return video_path.lower().startswith(_REMOTE_SCHEMES)

    def metadata(self) -> VideoMetadata:
        """Return metadata reported by the selected video stream."""
        if self.stream.average_rate is None:
            raise ValueError(f"Video stream has no average frame rate: {self.video_path}")

        return VideoMetadata(
            fps=float(self.stream.average_rate),
            frame_count=self.stream.frames,
            duration=self._duration(),
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

    def _duration(self) -> float:
        """Return the stream duration, falling back to the container's.

        Remote sources more often omit a per-stream duration than local files
        do, and the sampler cannot schedule windows without one.
        """
        if self.stream.duration is not None and self.stream.time_base is not None:
            return float(self.stream.duration * self.stream.time_base)
        if self.container.duration is not None:
            return float(self.container.duration) / av.time_base
        raise ValueError(f"Video stream has no duration: {self.video_path}")

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
