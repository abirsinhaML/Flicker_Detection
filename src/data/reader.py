"""Video decoding behind a small, backend-independent reader interface."""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterator, Sequence
from math import isfinite
from pathlib import Path

import av
import av.error
import numpy as np

from src.core.types import VideoMetadata, VideoWindow
from src.data.decode import DecodePolicy, FrameDecoder, SoftwareDecoder, build_decoder
from src.data.sampler import WindowSpec

logger = logging.getLogger(__name__)

# Below this, two windows are treated as touching rather than separated: window
# bounds come from float arithmetic on the stride, so an exact comparison would
# call a contiguous schedule gapped.
_GAP_TOLERANCE = 1e-9

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


def _has_gaps(specs: Sequence[WindowSpec]) -> bool:
    """Whether any moment between the first and last window goes uncovered."""
    return any(
        current.start_time - previous.end_time > _GAP_TOLERANCE
        for previous, current in zip(specs, specs[1:], strict=False)
    )


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

    def read_windows(
        self,
        specs: Sequence[WindowSpec],
        resize: tuple[int, int] | None = None,
    ) -> Iterator[VideoWindow]:
        """Decode a whole schedule, in time order, by the cheaper of two routes.

        A schedule that leaves no gaps -- the contiguous default -- is decoded in
        one forward pass: seek once, then hand each frame to whichever windows
        span it.  Seeking per window would instead re-decode from the preceding
        key frame every time, so the frames between that key frame and the window
        start are reconstructed and thrown away once per window.  Contiguous
        windows are ~200 per ten-minute video, which is ~200 of those.

        A schedule with gaps -- any stride larger than the window -- keeps the
        per-window seek, because walking the gaps would decode frames no window
        ever asks for.  The choice is made from the schedule rather than
        configured, so it cannot disagree with the sampler.

        Windows are yielded as they complete, so only the one or two currently
        open are held: a 3 s window of 320x180 is ~15 MB, and materializing all
        200 of a ten-minute video would not fit in a worker's share of memory.
        """
        if not specs:
            return
        self._validate_schedule(specs)
        target = resize if resize is not None else self.resize
        if _has_gaps(specs):
            logger.debug("Schedule has gaps; decoding %d windows by seek", len(specs))
            for spec in specs:
                yield self.read_window(
                    spec.start_time,
                    spec.end_time - spec.start_time,
                    resize,
                )
            return

        logger.debug("Schedule is gapless; decoding %d windows in one pass", len(specs))
        # A hardware failure part-way through must not cost the windows already
        # emitted, nor the ones not yet reached: downgrade and resume the walk
        # from the first window still owed.
        emitted = 0
        while emitted < len(specs):
            try:
                for window in self._walk(specs[emitted:], target):
                    emitted += 1
                    yield window
            except av.error.FFmpegError as error:
                if not self._can_downgrade():
                    raise
                logger.warning(
                    "Hardware decode failed (%s); resuming in software from window %d",
                    error,
                    emitted,
                )
                self._downgrade_to_software()
            else:
                return

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
        self._seek(start)

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

    def _walk(
        self,
        specs: Sequence[WindowSpec],
        target: tuple[int, int] | None,
    ) -> Iterator[VideoWindow]:
        """Yield every window in ``specs`` from a single forward decode.

        Each frame is converted once and appended to every window that spans it,
        so the overlap the end-anchored window creates costs no extra decode and
        no second copy -- the windows share the array.
        """
        self._seek(specs[0].start_time)

        # The decoder applies the scale itself when it was built for this size,
        # so re-scaling here would only undo the point of doing it on the GPU.
        needs_rescale = target is not None and self.decoder.output_size != target
        pending = deque(specs)
        open_windows: deque[tuple[WindowSpec, list[np.ndarray]]] = deque()

        for timestamp, frame in self.decoder.decode(self.container, self.stream):
            # Close first, so a frame is never appended to a window it postdates.
            # Ends are non-decreasing (validated), so the head closes first and
            # windows come out in time order.
            while open_windows and open_windows[0][0].end_time <= timestamp:
                spec, frames = open_windows.popleft()
                yield self._to_window(spec, frames, target)
            while pending and pending[0].start_time <= timestamp:
                open_windows.append((pending.popleft(), []))
            if not open_windows:
                if not pending:
                    return
                continue

            array = self._to_array(frame, target, needs_rescale)
            for spec, frames in open_windows:
                if spec.start_time <= timestamp < spec.end_time:
                    frames.append(array)

        # The stream ended before the schedule did.  Windows still open keep the
        # frames they got; ones never reached are reported empty rather than
        # dropped, so the schedule and the results stay the same length.
        for spec, frames in open_windows:
            yield self._to_window(spec, frames, target)
        for spec in pending:
            yield self._to_window(spec, [], target)

    def _to_array(
        self,
        frame: av.VideoFrame,
        target: tuple[int, int] | None,
        needs_rescale: bool,
    ) -> np.ndarray:
        if needs_rescale and target is not None:
            frame = frame.reformat(width=target[0], height=target[1], format="rgb24")
        return frame.to_ndarray(format="rgb24")

    def _to_window(
        self,
        spec: WindowSpec,
        frames: list[np.ndarray],
        target: tuple[int, int] | None,
    ) -> VideoWindow:
        frame_width, frame_height = target or (self.stream.width, self.stream.height)
        return VideoWindow(
            start_time=spec.start_time,
            end_time=spec.end_time,
            fps=self._fps(),
            frames=(
                np.stack(frames)
                if frames
                else np.empty((0, frame_height, frame_width, 3), dtype=np.uint8)
            ),
        )

    def _seek(self, start: float) -> None:
        """Position the container at the last key frame at or before ``start``.

        ``time_base`` is checked rather than assumed: a stream without one used to
        raise a bare TypeError from the division, which read as a decode bug
        rather than as the malformed stream it is.
        """
        if self.stream.time_base is None:
            raise ValueError(f"Video stream has no time base: {self.video_path}")
        self.container.seek(
            int(start / self.stream.time_base),
            stream=self.stream,
            backward=True,
            any_frame=False,
        )
        self.decoder.reset()

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

    @staticmethod
    def _validate_schedule(specs: Sequence[WindowSpec]) -> None:
        """Require a time-ordered schedule, which one forward pass depends on.

        Both bounds have to be non-decreasing: the single pass closes the
        earliest-ending window first, so an out-of-order schedule would emit
        windows out of time order and could close one before its frames arrived.
        Raising is better than silently sorting, because the caller pairs these
        windows with what it asked for.
        """
        for spec in specs:
            VideoReader._validate_window(
                spec.start_time,
                spec.end_time - spec.start_time,
                None,
            )
        for previous, current in zip(specs, specs[1:], strict=False):
            if current.start_time < previous.start_time or current.end_time < previous.end_time:
                raise ValueError(
                    "window schedule must be non-decreasing in both start and end time; "
                    f"{previous} precedes {current}"
                )
