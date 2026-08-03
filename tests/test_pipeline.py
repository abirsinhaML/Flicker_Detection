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

    def test_detector_columns_reconcile_with_flicker_score(self) -> None:
        """The weighted detector scores must reproduce the headline score.

        Reporting per-detector maxima across windows mixed evidence from
        different moments, so the columns could not explain the routing
        decision they were printed beside.
        """
        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "sample.mp4"
            self._write_video(video_path, flicker=True)

            config = load_config("configs/detector.yaml")
            result = process_video(video_path, config)

        weights = config["aggregation"]["weights"]
        self.assertEqual(set(result.detector_scores), set(weights))
        reconstructed = sum(weights[name] * score for name, score in result.detector_scores.items())
        self.assertAlmostEqual(reconstructed, result.flicker_score, places=6)

    def test_processes_manifest_into_fixed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "sample.mp4"
            manifest_path = temporary_path / "manifest.csv"
            output_path = temporary_path / "flags.csv"
            self._write_video(video_path)
            self._write_manifest(manifest_path, video_path)

            run_manifest(manifest_path, output_path, load_config("configs/detector.yaml"))

            rows = self._read_rows(output_path)
            self.assertEqual(tuple(rows[0]), OUTPUT_FIELDS)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["video_key"], "reference/sample.mp4")

    def test_resume_keeps_successes_and_retries_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "sample.mp4"
            manifest_path = temporary_path / "manifest.csv"
            output_path = temporary_path / "flags.csv"
            self._write_video(video_path)
            self._write_manifest(manifest_path, video_path)
            config = load_config("configs/detector.yaml")

            # A prior run that scored one video and failed another.
            with output_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writer.writerow({**_blank_row(), "status": "ok", "video_key": "already/done.mp4"})
                writer.writerow(
                    {
                        **_blank_row(),
                        "status": "error",
                        "video_key": "reference/sample.mp4",
                        "error": "ExpiredToken",
                    }
                )

            run_manifest(manifest_path, output_path, config, resume=True)

            rows = self._read_rows(output_path)
            by_key = {row["video_key"]: row for row in rows}
            # One row per key: the success is retained, the failure is retried.
            self.assertEqual(len(rows), 2)
            self.assertEqual(by_key["already/done.mp4"]["status"], "ok")
            self.assertEqual(by_key["reference/sample.mp4"]["status"], "ok")

    def test_resume_skips_work_when_everything_succeeded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "sample.mp4"
            manifest_path = temporary_path / "manifest.csv"
            output_path = temporary_path / "flags.csv"
            self._write_video(video_path)
            self._write_manifest(manifest_path, video_path)
            config = load_config("configs/detector.yaml")

            run_manifest(manifest_path, output_path, config)
            first = output_path.read_text(encoding="utf-8")
            run_manifest(manifest_path, output_path, config, resume=True)

            self.assertEqual(output_path.read_text(encoding="utf-8"), first)

    def test_limit_is_stable_across_resume(self) -> None:
        """--limit must address the same videos each run, not N further ones."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            video_path = temporary_path / "sample.mp4"
            manifest_path = temporary_path / "manifest.csv"
            output_path = temporary_path / "flags.csv"
            self._write_video(video_path)
            with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
                writer = csv.DictWriter(manifest_file, fieldnames=["key", "source_uri"])
                writer.writeheader()
                for index in range(4):
                    writer.writerow(
                        {"key": f"reference/{index}.mp4", "source_uri": str(video_path)}
                    )

            config = load_config("configs/detector.yaml")
            run_manifest(manifest_path, output_path, config, limit=2)
            first = self._read_rows(output_path)
            run_manifest(manifest_path, output_path, config, limit=2, resume=True)
            second = self._read_rows(output_path)

        # Rows land in completion order, so compare the work set, not the order.
        expected = {"reference/0.mp4", "reference/1.mp4"}
        self.assertEqual({row["video_key"] for row in first}, expected)
        self.assertEqual({row["video_key"] for row in second}, expected)
        self.assertEqual(len(second), 2)

    @staticmethod
    def _write_manifest(manifest_path: Path, video_path: Path) -> None:
        """Write a manifest addressing the video by durable identifier."""
        with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
            writer = csv.DictWriter(
                manifest_file,
                fieldnames=["key", "source_uri", "size_bytes", "last_modified"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "key": "reference/sample.mp4",
                    "source_uri": str(video_path),
                    "size_bytes": video_path.stat().st_size,
                    "last_modified": "2026-07-23T00:00:00",
                }
            )

    @staticmethod
    def _read_rows(output_path: Path) -> list[dict[str, str]]:
        with output_path.open(newline="", encoding="utf-8") as output_file:
            return list(csv.DictReader(output_file))

    @staticmethod
    def _write_video(video_path: Path, *, flicker: bool = False) -> None:
        """Write a tiny clip, optionally pulsing brightness frame to frame."""
        frame_count = 40 if flicker else 10
        output = av.open(str(video_path), mode="w")
        stream = output.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"

        for index in range(frame_count):
            level = 100 + (40 if flicker and index % 2 else 0)
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), level, dtype=np.uint8),
                format="rgb24",
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        output.close()


def _blank_row() -> dict[str, str]:
    return dict.fromkeys(OUTPUT_FIELDS, "")


if __name__ == "__main__":
    unittest.main()
