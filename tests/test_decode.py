"""Coverage for decoder selection and the hardware/software equivalence contract.

The tests that need a GPU skip themselves when there is none, so the suite stays
runnable offline and on CPU-only machines.  What is *not* optional is that the
software path keeps working and that a policy asking for hardware never loses a
video to the absence of it.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from src.data.decode import (
    CuvidDecoder,
    DecodePolicy,
    SoftwareDecoder,
    _hardware_obstacle,
    build_decoder,
)
from src.data.reader import VideoReader
from src.signals.preprocessing import SignalPreprocessor


def _write_video(video_path: Path, *, codec: str = "libx264", frames: int = 30) -> None:
    """Write a short clip with moving content, so rescaling is observable."""
    output = av.open(str(video_path), mode="w")
    stream = output.add_stream(codec, rate=10)
    stream.width = 128
    stream.height = 96
    stream.pix_fmt = "yuv420p"

    for index in range(frames):
        image = np.zeros((96, 128, 3), dtype=np.uint8)
        image[:, :, 0] = (index * 8) % 256
        image[index % 96, :, 1] = 255
        image[:, (index * 3) % 128, 2] = 255
        output.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")))
    output.mux(stream.encode())
    output.close()


def _cuda_available() -> bool:
    return _hardware_obstacle_for_h264() is None


def _hardware_obstacle_for_h264() -> str | None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "probe.mp4"
        _write_video(path)
        with av.open(str(path)) as container:
            return _hardware_obstacle(container.streams.video[0])


class DecodePolicyTests(unittest.TestCase):
    def test_defaults_to_auto_backend(self) -> None:
        self.assertEqual(DecodePolicy().backend, "auto")
        self.assertTrue(DecodePolicy().allow_fallback)

    def test_reads_the_config_block(self) -> None:
        policy = DecodePolicy.from_config(
            {"backend": "cpu", "gpu_index": 1, "threads": 4, "allow_fallback": False}
        )
        self.assertEqual(policy.backend, "cpu")
        self.assertEqual(policy.gpu_index, 1)
        self.assertEqual(policy.threads, 4)
        self.assertFalse(policy.allow_fallback)

    def test_missing_block_is_the_default_policy(self) -> None:
        self.assertEqual(DecodePolicy.from_config(None), DecodePolicy())

    def test_rejects_an_unknown_backend(self) -> None:
        with self.assertRaises(ValueError):
            DecodePolicy(backend="opencl")

    def test_rejects_a_negative_gpu_index(self) -> None:
        with self.assertRaises(ValueError):
            DecodePolicy(gpu_index=-1)


class DecoderSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.video_path = Path(self.directory.name) / "clip.mp4"
        _write_video(self.video_path)
        self.addCleanup(self.directory.cleanup)

    def test_cpu_backend_never_builds_a_hardware_decoder(self) -> None:
        with av.open(str(self.video_path)) as container:
            decoder = build_decoder(
                container.streams.video[0],
                DecodePolicy(backend="cpu"),
                resize=(64, 48),
            )
            self.assertIsInstance(decoder, SoftwareDecoder)
            self.assertEqual(decoder.name, "cpu")
            # Software decode cannot resize itself, so the reader must still do it.
            self.assertIsNone(decoder.output_size)

    def test_software_path_enables_frame_threading(self) -> None:
        with av.open(str(self.video_path)) as container:
            stream = container.streams.video[0]
            build_decoder(stream, DecodePolicy(backend="cpu"), resize=None)
            self.assertNotEqual(str(stream.thread_type), "NONE")

    def test_unsupported_codec_falls_back_rather_than_raising(self) -> None:
        path = Path(self.directory.name) / "clip_mpeg4.mp4"
        _write_video(path, codec="mpeg4")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            # Whatever this machine supports, an auto policy must yield a
            # working decoder rather than propagate a hardware problem.
            decoder = build_decoder(stream, DecodePolicy(), resize=(64, 48))
            self.assertIn(decoder.name, {"cpu", "cuda"})

    def test_oversized_stream_is_refused_by_the_hardware_probe(self) -> None:
        class _Stub:
            width = 9000
            height = 9000

            class codec_context:  # noqa: N801 - mirrors the PyAV attribute name
                name = "h264"

        self.assertIn("exceeds the NVDEC", str(_hardware_obstacle(_Stub())))

    def test_probe_rejects_a_codec_nvdec_does_not_implement(self) -> None:
        class _Stub:
            width = 640
            height = 480

            class codec_context:  # noqa: N801
                name = "prores"

        self.assertIn("no NVDEC decoder", str(_hardware_obstacle(_Stub())))


class ReaderBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.video_path = Path(self.directory.name) / "clip.mp4"
        _write_video(self.video_path)
        self.addCleanup(self.directory.cleanup)

    def test_constructor_resize_is_applied_without_a_per_call_argument(self) -> None:
        with VideoReader(
            self.video_path, decode_policy=DecodePolicy(backend="cpu"), resize=(64, 48)
        ) as reader:
            window = reader.read_window(0.0, 1.0)
        self.assertEqual(window.frames.shape[1:], (48, 64, 3))

    def test_per_call_resize_still_overrides_the_reader_default(self) -> None:
        with VideoReader(
            self.video_path, decode_policy=DecodePolicy(backend="cpu"), resize=(64, 48)
        ) as reader:
            window = reader.read_window(0.0, 1.0, resize=(32, 24))
        self.assertEqual(window.frames.shape[1:], (24, 32, 3))

    def test_reports_the_backend_actually_in_use(self) -> None:
        with VideoReader(self.video_path, decode_policy=DecodePolicy(backend="cpu")) as reader:
            self.assertEqual(reader.decode_backend, "cpu")

    @unittest.skipUnless(_cuda_available(), "no NVDEC-capable GPU on this machine")
    def test_hardware_and_software_agree_on_the_measured_signal(self) -> None:
        """The two backends must be interchangeable for calibration to survive.

        They are not bit-identical -- NVDEC's scaler is not swscale's -- so the
        contract is on the signal the detectors actually consume, not on pixels.
        """
        resize = (64, 48)
        with VideoReader(
            self.video_path, decode_policy=DecodePolicy(backend="cpu"), resize=resize
        ) as reader:
            software = reader.read_window(0.0, 2.0)
        with VideoReader(
            self.video_path, decode_policy=DecodePolicy(backend="cuda"), resize=resize
        ) as reader:
            self.assertEqual(reader.decode_backend, "cuda")
            hardware = reader.read_window(0.0, 2.0)

        self.assertEqual(software.frames.shape, hardware.frames.shape)
        software_luma = software.frames.astype(np.float32).mean(axis=(1, 2, 3))
        hardware_luma = hardware.frames.astype(np.float32).mean(axis=(1, 2, 3))
        # One level out of 255 is well inside the tolerance any fitted scale has.
        self.assertLess(float(np.abs(software_luma - hardware_luma).max()), 1.0)

    @unittest.skipUnless(_cuda_available(), "no NVDEC-capable GPU on this machine")
    def test_final_window_is_not_truncated_by_the_hardware_decoder(self) -> None:
        """The end-anchored last window must not lose the decoder's held frames.

        A hardware decoder holds frames back for reordering and only releases
        them when the flush packet arrives.  Ignoring that packet drops the tail
        of the stream -- and :class:`WindowSampler` anchors a window to the end
        of *every* video, so the loss would land on every result rather than on
        an obvious edge case.
        """
        with VideoReader(
            self.video_path, decode_policy=DecodePolicy(backend="cpu"), resize=(64, 48)
        ) as reader:
            duration = reader.metadata().duration
            expected = len(reader.read_window(max(duration - 1.0, 0.0), 1.0).frames)

        with VideoReader(
            self.video_path,
            decode_policy=DecodePolicy(backend="cuda", allow_fallback=False),
            resize=(64, 48),
        ) as reader:
            actual = len(reader.read_window(max(duration - 1.0, 0.0), 1.0).frames)

        self.assertEqual(actual, expected)
        self.assertGreater(expected, 0)

    @unittest.skipUnless(_cuda_available(), "no NVDEC-capable GPU on this machine")
    def test_hardware_decoder_resizes_inside_the_decoder(self) -> None:
        with av.open(str(self.video_path)) as container:
            decoder = build_decoder(
                container.streams.video[0],
                DecodePolicy(backend="cuda", gpu_resize=True),
                resize=(64, 48),
            )
            self.assertIsInstance(decoder, CuvidDecoder)
            self.assertEqual(decoder.output_size, (64, 48))
            decoder.close()


class BatchedColourConversionTests(unittest.TestCase):
    """The batched conversion is only safe if it is bit-identical."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.frames = rng.integers(0, 256, (7, 12, 20, 3), dtype=np.uint8)

    def test_yuv_matches_per_frame_conversion_exactly(self) -> None:
        import cv2

        expected = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YUV) for f in self.frames])
        np.testing.assert_array_equal(SignalPreprocessor.rgb_to_yuv(self.frames), expected)

    def test_lab_matches_per_frame_conversion_exactly(self) -> None:
        import cv2

        expected = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2LAB) for f in self.frames])
        np.testing.assert_array_equal(SignalPreprocessor.rgb_to_lab(self.frames), expected)

    def test_empty_batch_is_preserved(self) -> None:
        empty = np.empty((0, 12, 20, 3), dtype=np.uint8)
        self.assertEqual(SignalPreprocessor.rgb_to_yuv(empty).shape, empty.shape)

    def test_non_contiguous_input_is_handled(self) -> None:
        import cv2

        view = self.frames[::2]
        expected = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YUV) for f in view])
        np.testing.assert_array_equal(SignalPreprocessor.rgb_to_yuv(view), expected)


if __name__ == "__main__":
    unittest.main()
