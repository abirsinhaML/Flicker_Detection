"""Tests for analysis-region selection on fisheye framing."""

from __future__ import annotations

import unittest

import numpy as np

from src.core.types import VideoWindow
from src.signals.extractor import SignalExtractor
from src.signals.mask import AnalysisRegion, RegionPolicy

# The corpus geometry: a 960x720 image circle pillarboxed in 1280x720, scaled
# down to the analysed resolution.
WIDTH, HEIGHT = 320, 180
BAR = 40


def _pillarboxed(frames: int = 12, level: float = 100.0, swing: float = 0.0) -> np.ndarray:
    """Luma frames with matte bars left and right, optionally pulsing."""
    batch = np.zeros((frames, HEIGHT, WIDTH), dtype=np.float32)
    for index in range(frames):
        batch[index, :, BAR : WIDTH - BAR] = level + (swing if index % 2 else -swing)
    return batch


class AnalysisRegionTests(unittest.TestCase):
    def test_finds_the_image_circle_and_drops_matte_bars(self) -> None:
        region = AnalysisRegion.detect(_pillarboxed(), RegionPolicy())

        self.assertFalse(region.mask[:, :BAR].any())
        self.assertFalse(region.mask[:, WIDTH - BAR :].any())
        self.assertTrue(region.mask[:, BAR : WIDTH - BAR].all())
        self.assertAlmostEqual(region.valid_fraction, (WIDTH - 2 * BAR) / WIDTH, places=6)

    def test_judges_deadness_on_peak_not_mean_brightness(self) -> None:
        """A scene region that is dark on average but brightens is not dead."""
        frames = _pillarboxed()
        frames[:, :20, BAR : WIDTH - BAR] = 1.0
        frames[6, :20, BAR : WIDTH - BAR] = 200.0

        region = AnalysisRegion.detect(frames, RegionPolicy(border_threshold=8.0))

        self.assertTrue(region.mask[:20, BAR : WIDTH - BAR].all())

    def test_excludes_the_wearers_body_from_the_lower_frame(self) -> None:
        region = AnalysisRegion.detect(
            _pillarboxed(),
            RegionPolicy(exclude_bottom_fraction=0.25),
        )

        kept = int(round(HEIGHT * 0.75))
        self.assertTrue(region.mask[:kept, BAR : WIDTH - BAR].all())
        self.assertFalse(region.mask[kept:, :].any())
        self.assertTrue(region.rows.max() < kept)

    def test_falls_back_to_the_whole_frame_when_almost_nothing_survives(self) -> None:
        dark = np.full((8, HEIGHT, WIDTH), 2.0, dtype=np.float32)

        with self.assertLogs("src.signals.mask", level="WARNING"):
            region = AnalysisRegion.detect(dark, RegionPolicy(border_threshold=8.0))

        self.assertTrue(region.mask.all())
        self.assertEqual(region.valid_fraction, 1.0)

    def test_drops_rows_too_narrow_to_average(self) -> None:
        """Rows grazing the image circle are mostly noise, not signal."""
        circular = np.zeros((6, HEIGHT, WIDTH), dtype=np.float32)
        centre_y, centre_x = HEIGHT / 2, WIDTH / 2
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
        inside = ((yy - centre_y) / centre_y) ** 2 + ((xx - centre_x) / centre_x) ** 2 <= 1.0
        circular[:, inside] = 120.0

        region = AnalysisRegion.detect(circular, RegionPolicy())

        self.assertNotIn(0, region.rows, "the top row spans almost no live pixels")
        self.assertIn(HEIGHT // 2, region.rows)

    def test_rejects_malformed_input(self) -> None:
        with self.assertRaises(ValueError):
            AnalysisRegion.detect(np.zeros((4, 4), dtype=np.float32), RegionPolicy())
        with self.assertRaises(ValueError):
            AnalysisRegion.detect(np.empty((0, 4, 4), dtype=np.float32), RegionPolicy())

    def test_policy_validates_its_own_bounds(self) -> None:
        for kwargs in (
            {"border_threshold": -1.0},
            {"exclude_bottom_fraction": 1.0},
            {"min_valid_fraction": 0.0},
        ):
            with self.assertRaises(ValueError):
                RegionPolicy(**kwargs)  # type: ignore[arg-type]

    def test_policy_reads_the_config_block(self) -> None:
        policy = RegionPolicy.from_config(
            {"border_threshold": 4.0, "exclude_bottom_fraction": 0.3, "min_valid_fraction": 0.5}
        )
        self.assertEqual(policy.border_threshold, 4.0)
        self.assertEqual(policy.exclude_bottom_fraction, 0.3)
        self.assertEqual(RegionPolicy.from_config(None).exclude_bottom_fraction, 0.0)


class MaskedSignalTests(unittest.TestCase):
    """The property that makes the pillarbox harmless to severity scoring."""

    def _window(self, swing: float) -> VideoWindow:
        luma = _pillarboxed(frames=16, level=100.0, swing=swing)
        rgb = np.repeat(luma[..., None], 3, axis=3).clip(0, 255).astype(np.uint8)
        return VideoWindow(start_time=0.0, end_time=0.5, fps=30.0, frames=rgb)

    @staticmethod
    def _luma(window: VideoWindow, region: AnalysisRegion) -> np.ndarray:
        from src.signals.luminance import LuminanceExtractor
        from src.signals.preprocessing import SignalPreprocessor

        yuv = SignalPreprocessor.rgb_to_yuv(window.frames)
        return LuminanceExtractor.global_signal(yuv, region)

    def test_modulation_depth_is_unchanged_by_the_matte_bars(self) -> None:
        """Why the pillarbox never corrupted severity: the bars cancel in a ratio.

        They scale the temporal mean and its fluctuation by the same crop
        fraction, so the coefficient of variation is invariant.
        """
        from src.features.frequency import FrequencyAnalyzer

        window = self._window(swing=6.0)
        masked = SignalExtractor.extract(window, RegionPolicy()).luma
        whole = self._luma(window, AnalysisRegion.full_frame(HEIGHT, WIDTH))

        self.assertAlmostEqual(
            FrequencyAnalyzer.modulation_depth(masked),
            FrequencyAnalyzer.modulation_depth(whole),
            places=4,
        )

    def test_absolute_luma_level_is_restored_by_masking(self) -> None:
        """And why absolute measurements did need fixing: they do not cancel."""
        window = self._window(swing=0.0)
        masked = SignalExtractor.extract(window, RegionPolicy()).luma
        whole = self._luma(window, AnalysisRegion.full_frame(HEIGHT, WIDTH))

        self.assertAlmostEqual(float(masked.mean()), 100.0, delta=1.0)
        # Diluted by the 25% dead fraction when the bars are included.
        self.assertAlmostEqual(float(whole.mean()), 75.0, delta=1.0)

    def test_column_band_profiles_cover_the_live_width(self) -> None:
        signals = SignalExtractor.extract(self._window(swing=6.0), RegionPolicy())

        self.assertEqual(signals.column_band_profiles.shape[0], 3)
        self.assertEqual(
            signals.column_band_profiles.shape[2],
            signals.row_profiles.shape[1],
        )


if __name__ == "__main__":
    unittest.main()
