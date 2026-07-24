"""End-to-end coverage for the single-video MVP."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np

from main import OUTPUT_FIELDS, load_config, process_video, run_manifest, write_result_csv


class PipelineTests(unittest.TestCase):
    """Verify decoding, sampling, detection, aggregation, and CSV output."""

    def test_processes_one_video_and_writes_one_csv_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "sample.mp4"
            output_path = temporary_path / "result.csv"
            self._write_video(video_path)

            config = load_config(Path("configs/detector.yaml"))
            result = process_video(video_path, config)
            write_result_csv(result, output_path)

            self.assertEqual(result.video_key, "sample.mp4")
            self.assertIn("IlluminantDetector", result.detector_scores)
            self.assertIn("RollingBandDetector", result.detector_scores)
            self.assertTrue(output_path.exists())

            with output_path.open(newline="", encoding="utf-8") as output_file:
                rows = list(csv.DictReader(output_file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_key"], "sample.mp4")

    def test_processes_manifest_into_fixed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "sample.mp4"
            manifest_path = temporary_path / "manifest.csv"
            output_path = temporary_path / "flags.csv"
            self._write_video(video_path)
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(
                    manifest_file,
                    fieldnames=[
                        "key",
                        "size_bytes",
                        "last_modified",
                        "presigned_url_7day",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "key": "reference/sample.mp4",
                        "size_bytes": video_path.stat().st_size,
                        "last_modified": "2026-07-23T00:00:00",
                        "presigned_url_7day": str(video_path),
                    }
                )

            run_manifest(manifest_path, output_path, load_config("configs/detector.yaml"))

            with output_path.open(newline="", encoding="utf-8") as output_file:
                rows = list(csv.DictReader(output_file))
            self.assertEqual(tuple(rows[0]), OUTPUT_FIELDS)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["video_key"], "reference/sample.mp4")

    @staticmethod
    def _write_video(video_path: Path) -> None:
        output = av.open(str(video_path), mode="w")
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"

        for _ in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), 100, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()


if __name__ == "__main__":
    unittest.main()
