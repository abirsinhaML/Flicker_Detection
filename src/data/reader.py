"""Video decoding behind a small, backend-independent reader interface."""

from __future__ import annotations

import logging
from math import isfinite
from pathlib import Path

import av
import av.error
import numpy as np

from src.core.types import VideoMetadata, VideoWindow
from src.data.decode import DecodePolicy, FrameDecoder, SoftwareDecoder, build_decoder

logger = logging.getLogger(__name__)

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

    ``resize`` given here is the target frame size for every window.  Passing it
    to the constructor rather than to each :meth:`read_window` call is what lets
    a hardware decoder apply the scale itself, so full-resolution frames are
    never copied out of the GPU.  ``read_window`` still accepts a per-call
    ``resize`` for callers that need a different size, at the cost of a software
    rescale on top.
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        decode_policy: DecodePolicy | None = None,
        resize: tuple[int, int] | None = None,
    ) -> None:
        self.video_path = str(video_path)
        self.resize = resize
        self.decode_policy = decode_policy or DecodePolicy()
        self.container = (
            av.open(self.video_path, options=dict(_HTTP_OPTIONS), timeout=_REMOTE_TIMEOUT)
            if self.is_remote(self.video_path)
            else av.open(self.video_path)
        )

        if not self.container.streams.video:
            self.container.close()
            raise ValueError(f"No video stream found in: {self.video_path}")

        self.stream = self.container.streams.video[0]
        self.decoder: FrameDecoder = build_decoder(
            self.stream,
            self.decode_policy,
            resize=resize,
            source=self.video_path,
        )

    @staticmethod
    def is_remote(video_path: str) -> bool:
        """Return whether FFmpeg will read this source over HTTP."""
        return video_path.lower().startswith(_REMOTE_SCHEMES)

    @property
    def decode_backend(self) -> str:
        """Which backend is actually decoding, after any fallback."""
        return self.decoder.name

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
        is ``(width, height)``; it defaults to the reader's own and is applied by
        the decoder itself where the backend supports it.
        """
        self._validate_window(start, duration, resize)
        target = resize if resize is not None else self.resize
        end = start + duration

        try:
            frames = self._decode_window(start, end, target)
        except av.error.FFmpegError as error:
            # A hardware decoder can fail on a stream that probed as supported:
            # an unexpected profile, exhausted GPU memory, or one NVDEC session
            # too many.  Retry the window in software rather than lose the video.
            if not self._can_downgrade():
                raise
            logger.warning("Hardware decode failed (%s); retrying window in software", error)
            self._downgrade_to_software()
            frames = self._decode_window(start, end, target)

        frame_width, frame_height = target or (self.stream.width, self.stream.height)
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
        self.decoder.close()
        self.container.close()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def _decode_window(
        self,
        start: float,
        end: float,
        target: tuple[int, int] | None,
    ) -> list[np.ndarray]:
        """Collect the RGB frames of one window from the active decoder."""
        self.container.seek(
            int(start / self.stream.time_base),
            stream=self.stream,
            backward=True,
            any_frame=False,
        )
        self.decoder.reset()

        # The decoder applies the scale itself when it was built for this size,
        # so re-scaling here would only undo the point of doing it on the GPU.
        needs_rescale = target is not None and self.decoder.output_size != target

        frames: list[np.ndarray] = []
        for timestamp, frame in self.decoder.decode(self.container, self.stream):
            if timestamp < start:
                continue
            if timestamp >= end:
                break

            if needs_rescale and target is not None:
                frame = frame.reformat(width=target[0], height=target[1], format="rgb24")

            frames.append(frame.to_ndarray(format="rgb24"))
        return frames

    def _can_downgrade(self) -> bool:
        return self.decoder.name != "cpu" and self.decode_policy.allow_fallback

    def _downgrade_to_software(self) -> None:
        self.decoder.close()
        self.decoder = SoftwareDecoder(self.stream, self.decode_policy.threads)

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
