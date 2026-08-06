"""Decoder selection: NVDEC hardware decode when it is available, software otherwise.

Decoding is this pipeline's whole cost.  The analysed signal is a 320x180 window
of a few seconds, but the sources are 3840x2880 at 29.97 fps, so libavcodec
spends orders of magnitude more CPU reconstructing frames than the detectors
spend measuring them: about 4.1 s of CPU per 3s window against 0.24 s for every
detector and feature combined.  NVDEC decoding the same window with its own
downscale costs 0.54 s, a 7.6x reduction in the part that dominates.

Two properties of the hardware path matter beyond raw speed:

* The scale happens *inside* the decoder (``resize``), so full-resolution frames
  never cross PCIe and swscale never runs.  Only the 320x180 result is copied
  back, 173 KB per frame.
* NVDEC is a fixed-function engine, separate from the SMs.  It offloads the work
  rather than merely moving it, which is what frees cores for more concurrent
  videos.

Hardware decode is an optimisation, never a requirement: every failure mode --
no driver, no CUDA device, an unsupported codec or profile, exhausted GPU
memory, too many concurrent NVDEC sessions -- degrades to the software decoder
and is logged once, because a batch that silently skips videos is worse than a
slow one.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import av
import av.error

logger = logging.getLogger(__name__)

# FFmpeg decoder name per codec, for codecs NVDEC implements.  A codec absent
# from this table is decoded in software rather than guessed at.
_CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "vc1": "vc1_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
    "mjpeg": "mjpeg_cuvid",
}

# NVDEC's maximum decodable dimension per codec.  This corpus is 3840x2880,
# comfortably inside H.264's limit, but a larger source must not be handed to a
# decoder that will reject it mid-batch.
_MAX_DIMENSION = {
    "h264": 4096,
    "mpeg4": 4080,
    "mpeg2video": 4080,
    "vc1": 4080,
    "vp8": 4096,
}
_DEFAULT_MAX_DIMENSION = 8192

BACKENDS = ("auto", "cuda", "cpu")


@dataclass(frozen=True, slots=True)
class DecodePolicy:
    """How a window's frames are turned back into pixels.

    ``backend`` is ``auto`` (hardware when usable, software otherwise), ``cuda``
    (hardware, still falling back on failure unless ``allow_fallback`` is off),
    or ``cpu``.  ``threads`` applies only to the software path, where ``0`` lets
    FFmpeg choose; frame threading is what makes software 4K decode tolerable
    and is off by default in PyAV.

    ``gpu_resize`` decides where the 3840x2880 -> 320x180 downscale happens, and
    it trades throughput against exactness rather than being a pure win.  Per 3s
    window of 4K, measured on this corpus's frame size:

        software decode + swscale        4.08 s CPU
        NVDEC + swscale                  2.24 s CPU
        NVDEC + in-decoder downscale     0.54 s CPU

    Off, the downscale stays in swscale and the result is bit-identical to the
    software path wherever chroma is trivial -- NVDEC implements the same
    normative H.264 reconstruction.  What residual there is comes from NV12 and
    YUV420P taking different swscale chroma paths, not from the decode.

    On, NVDEC also resamples, which its own scaler does differently from
    swscale.  Measured against the software path: no change worth reporting on a
    clip carrying a genuine rolling band (``flicker_score`` +0.0013, worst
    channel 0.0077), against decision boundaries at 0.328 and 0.51.  The drift
    is largest where a detector has no real signal to measure and is reporting
    its own noise floor.  Validate on real footage with
    ``scripts/compare_decode_backends.py`` before turning it on for a batch.
    """

    backend: str = "auto"
    gpu_index: int = 0
    threads: int = 0
    allow_fallback: bool = True
    gpu_resize: bool = False

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"decode.backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.gpu_index < 0:
            raise ValueError("decode.gpu_index must be non-negative")
        if self.threads < 0:
            raise ValueError("decode.threads must be non-negative")

    @classmethod
    def from_config(cls, config: dict[str, object] | None) -> DecodePolicy:
        """Build a policy from the ``decode`` config block."""
        section = config or {}
        return cls(
            backend=str(section.get("backend", "auto")),
            gpu_index=int(section.get("gpu_index", 0)),  # type: ignore[call-overload]
            threads=int(section.get("threads", 0)),  # type: ignore[call-overload]
            allow_fallback=bool(section.get("allow_fallback", True)),
            gpu_resize=bool(section.get("gpu_resize", False)),
        )


class FrameDecoder(Protocol):
    """Turns demuxed packets of one stream back into frames."""

    name: str
    # Frame size the decoder already emits, when it applies the scale itself.
    output_size: tuple[int, int] | None

    def reset(self) -> None:
        """Discard decoder state, as required after a seek."""

    def decode(
        self, container: av.container.InputContainer, stream: av.video.stream.VideoStream
    ) -> Iterator[tuple[float, av.VideoFrame]]:
        """Yield ``(timestamp_seconds, frame)`` from the current position."""

    def close(self) -> None:
        """Release any decoder resources."""


class SoftwareDecoder:
    """libavcodec software decode, with frame threading enabled.

    PyAV leaves ``thread_type`` unset, which decodes single-threaded.  On 4K
    sources that is the difference between 22 s and 8 s per 30 s of video.
    """

    name = "cpu"

    def __init__(self, stream: av.video.stream.VideoStream, threads: int = 0) -> None:
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = threads
        self.output_size: tuple[int, int] | None = None

    def reset(self) -> None:
        # ``container.decode`` re-primes the stream's own context after a seek.
        pass

    def decode(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
    ) -> Iterator[tuple[float, av.VideoFrame]]:
        for frame in container.decode(stream):
            timestamp = _frame_time(frame, stream)
            if timestamp is not None:
                yield timestamp, frame

    def close(self) -> None:
        pass


class CuvidDecoder:
    """NVDEC decode with the downscale performed inside the decoder.

    Frames are demuxed here and pushed through a standalone decoder context
    rather than ``container.decode``, because only a context we own can carry
    the ``resize`` option.  Without it the GPU would hand back full-resolution
    frames and the copy back to host would cost more than it saved.

    Such a context has no ``time_base`` of its own -- setting one on a decoder
    raises -- so frame timing is reconstructed from the packet timestamps and
    the stream's own time base.
    """

    name = "cuda"

    def __init__(
        self,
        stream: av.video.stream.VideoStream,
        *,
        resize: tuple[int, int] | None,
        gpu_index: int = 0,
    ) -> None:
        decoder_name = _CUVID_DECODERS[stream.codec_context.name]
        codec = av.codec.Codec(decoder_name, "r")
        context = codec.create("video")
        context.extradata = stream.codec_context.extradata

        options = {"gpu": str(gpu_index)}
        if resize is not None:
            options["resize"] = f"{resize[0]}x{resize[1]}"
        context.options = options

        self.context = context
        self.output_size = resize

    def reset(self) -> None:
        self.context.flush_buffers()

    def decode(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
    ) -> Iterator[tuple[float, av.VideoFrame]]:
        for packet in container.demux(stream):
            # The demuxer's final packet carries no dts: it is the flush packet,
            # and feeding it to the decoder drains the frames still held back for
            # reordering.  Skipping it silently truncates the tail of the stream,
            # which is precisely the end-anchored final window that the sampler
            # emits for every video.
            for frame in self.context.decode(packet):
                timestamp = _frame_time(frame, stream)
                if timestamp is not None:
                    yield timestamp, frame
            if packet.dts is None:
                break

    def close(self) -> None:
        # PyAV exposes no explicit close on a decoder context; flushing releases
        # the NVDEC surfaces, and dropping the reference frees the CUDA context.
        # Closing must not be what fails a video that already decoded fine.
        with contextlib.suppress(av.error.FFmpegError, RuntimeError, ValueError):
            self.context.flush_buffers()


def build_decoder(
    stream: av.video.stream.VideoStream,
    policy: DecodePolicy,
    *,
    resize: tuple[int, int] | None,
    source: str = "",
) -> FrameDecoder:
    """Return the best decoder this policy and this stream allow.

    Selection never raises for want of hardware.  When ``cuda`` is asked for and
    cannot be provided, the reason is logged and the software decoder is
    returned instead, unless the policy forbids that.
    """
    if policy.backend == "cpu":
        return SoftwareDecoder(stream, policy.threads)

    unusable = _hardware_obstacle(stream)
    if unusable is None:
        try:
            return CuvidDecoder(
                stream,
                resize=resize if policy.gpu_resize else None,
                gpu_index=policy.gpu_index,
            )
        except (av.error.FFmpegError, ValueError, RuntimeError) as error:
            unusable = f"NVDEC decoder could not be created ({error})"

    if not policy.allow_fallback:
        raise RuntimeError(f"Hardware decode unavailable and fallback disabled: {unusable}")

    _warn_once(unusable, source)
    return SoftwareDecoder(stream, policy.threads)


def _hardware_obstacle(stream: av.video.stream.VideoStream) -> str | None:
    """Return why NVDEC cannot decode this stream, or ``None`` if it can."""
    codec_name = stream.codec_context.name
    decoder_name = _CUVID_DECODERS.get(codec_name)
    if decoder_name is None:
        return f"codec {codec_name!r} has no NVDEC decoder"

    try:
        av.codec.Codec(decoder_name, "r")
    except Exception:
        return f"{decoder_name} is not built into this FFmpeg"

    if "cuda" not in av.codec.hwaccel.hwdevices_available():
        return "no CUDA hardware device is available"

    limit = _MAX_DIMENSION.get(codec_name, _DEFAULT_MAX_DIMENSION)
    if max(stream.width or 0, stream.height or 0) > limit:
        return f"{stream.width}x{stream.height} exceeds the NVDEC {codec_name} limit of {limit}"

    return None


def _frame_time(frame: av.VideoFrame, stream: av.video.stream.VideoStream) -> float | None:
    """Presentation time in seconds, for frames that carry a usable timestamp.

    A decoder context created by hand has no time base, so ``frame.time`` is
    ``None`` on the hardware path and the stream's time base supplies it.
    """
    if frame.time is not None:
        return float(frame.time)
    if frame.pts is not None and stream.time_base is not None:
        return float(frame.pts * stream.time_base)
    return None


_warned: set[str] = set()


def _warn_once(reason: str, source: str) -> None:
    """Log a hardware-decode fallback once per reason, per process.

    A corpus of a quarter million videos would otherwise repeat one line per
    video and bury everything else in the log.
    """
    if reason in _warned:
        logger.debug("Software decode for %s: %s", source, reason)
        return
    _warned.add(reason)
    logger.warning("Falling back to software decode: %s", reason)
