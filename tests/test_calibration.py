"""Tests for deterministic threshold fitting and report generation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.calibration.evaluation import calibrate, write_calibration_report


class CalibrationTests(unittest.TestCase):
    def test_fits_thresholds_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            scores_path = directory / "scores.csv"
            labels_path = directory / "labels.csv"
            report_path = directory / "report.md"
            self._write_fixture(scores_path, labels_path)

            result = calibrate(scores_path, labels_path)
            write_calibration_report(result, report_path)

            self.assertLess(result.mild_threshold, result.extreme_threshold)
            self.assertGreater(result.train_size, 0)
            self.assertGreater(result.validation_size, 0)
            self.assertTrue(report_path.exists())
            self.assertIn("Held-out evaluation", report_path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_fixture(scores_path: Path, labels_path: Path) -> None:
        with (
            scores_path.open("w", newline="", encoding="utf-8") as scores_file,
            labels_path.open("w", newline="", encoding="utf-8") as labels_file,
        ):
            score_writer = csv.DictWriter(
                scores_file, fieldnames=["video_key", "flicker_score", "status"]
            )
            label_writer = csv.DictWriter(labels_file, fieldnames=["video_key", "label"])
            score_writer.writeheader()
            label_writer.writeheader()
            for index in range(60):
                if index % 3 == 0:
                    score, label = 0.05, "none"
                elif index % 3 == 1:
                    score, label = 0.45, "mild"
                else:
                    score, label = 0.90, "extreme"
                key = f"sample-{index:03d}.mp4"
                score_writer.writerow({"video_key": key, "flicker_score": score, "status": "ok"})
                label_writer.writerow({"video_key": key, "label": label})
