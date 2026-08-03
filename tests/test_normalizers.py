"""Tests for the calibration layer that grades raw detector measurements."""

from __future__ import annotations

import unittest

from src.calibration.thresholds import (
    AWBNormalizer,
    IlluminantNormalizer,
    RollingBandNormalizer,
    soft_saturate,
)
from src.detectors.awb import AWBMetrics
from src.detectors.illuminant import IlluminantMetrics
from src.detectors.rolling_band import RollingBandMetrics


def _awb_metrics(**overrides: float | bool) -> AWBMetrics:
    defaults: dict[str, float | bool] = {
        "chroma_prominence": 0.0,
        "chroma_periodicity": 0.0,
        "chroma_variance": 0.0,
        "luma_chroma_decorrelation": 0.5,
        "ae_prominence": 0.0,
        "ae_periodicity": 0.0,
        "ae_modulation_depth": 0.0,
        "ae_dominant_frequency": 0.0,
        "ae_in_band": False,
    }
    return AWBMetrics(**{**defaults, **overrides})  # type: ignore[arg-type]


def _illuminant_metrics(**overrides: float) -> IlluminantMetrics:
    defaults: dict[str, float] = {
        "dominant_frequency": 0.0,
        "dominant_power": 0.0,
        "peak_prominence": 0.0,
        "peak_to_median_ratio": 0.0,
        "modulation_depth": 0.0,
    }
    return IlluminantMetrics(**{**defaults, **overrides})


class SoftSaturateTests(unittest.TestCase):
    def test_reference_is_the_half_way_point(self) -> None:
        self.assertAlmostEqual(soft_saturate(4.0, 4.0), 0.5)

    def test_is_bounded_and_never_reaches_one(self) -> None:
        self.assertEqual(soft_saturate(0.0, 4.0), 0.0)
        self.assertLess(soft_saturate(1e9, 4.0), 1.0)
        self.assertGreater(soft_saturate(1e9, 4.0), 0.99)

    def test_preserves_ordering_far_above_the_reference(self) -> None:
        self.assertLess(soft_saturate(100.0, 4.0), soft_saturate(1000.0, 4.0))

    def test_treats_negative_and_non_finite_input_as_no_evidence(self) -> None:
        self.assertEqual(soft_saturate(-1.0, 4.0), 0.0)
        self.assertEqual(soft_saturate(float("nan"), 4.0), 0.0)


class IlluminantNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = IlluminantNormalizer(
            prominence_reference=4.0,
            periodicity_reference=8.0,
            modulation_depth_reference=0.02,
        )

    def test_every_component_is_half_at_its_reference(self) -> None:
        """Periodicity 0.5 times severity 0.5, since the two are multiplied."""
        score = self.normalizer(
            _illuminant_metrics(
                dominant_frequency=7.0,
                peak_prominence=4.0,
                peak_to_median_ratio=8.0,
                modulation_depth=0.02,
            )
        )
        self.assertAlmostEqual(score, 0.25)

    def test_periodicity_without_severity_scores_near_zero(self) -> None:
        """A textbook-clean spectral peak with a negligible swing is invisible."""
        score = self.normalizer(
            _illuminant_metrics(
                dominant_frequency=100.0,
                peak_prominence=1e4,
                peak_to_median_ratio=1e4,
                modulation_depth=1e-5,
            )
        )
        self.assertLess(score, 0.01)

    def test_severity_without_periodicity_scores_near_zero(self) -> None:
        """A large aperiodic swing is motion or a scene change, not flicker."""
        score = self.normalizer(_illuminant_metrics(modulation_depth=0.5))
        self.assertLess(score, 0.01)

    def test_rejects_non_positive_references(self) -> None:
        with self.assertRaises(ValueError):
            IlluminantNormalizer(
                prominence_reference=0.0,
                periodicity_reference=8.0,
                modulation_depth_reference=0.02,
            )

    def test_rejects_foreign_metrics(self) -> None:
        with self.assertRaises(TypeError):
            self.normalizer(_awb_metrics())


class AWBNormalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.normalizer = AWBNormalizer(
            chroma_prominence_reference=2.0,
            chroma_periodicity_reference=4.0,
            chroma_variance_reference=0.5,
            ae_prominence_reference=2.0,
            ae_periodicity_reference=4.0,
            ae_modulation_depth_reference=0.02,
        )

    def test_no_evidence_scores_zero(self) -> None:
        self.assertEqual(self.normalizer(_awb_metrics()), 0.0)

    def test_ae_evidence_only_counts_inside_the_band(self) -> None:
        strong = {"ae_prominence": 20.0, "ae_periodicity": 40.0, "ae_modulation_depth": 0.5}
        in_band = self.normalizer(
            _awb_metrics(**strong, ae_in_band=True, ae_dominant_frequency=2.0)
        )
        out_of_band = self.normalizer(
            _awb_metrics(**strong, ae_in_band=False, ae_dominant_frequency=100.0)
        )

        self.assertGreater(in_band, 0.8)
        self.assertEqual(out_of_band, 0.0)

    def test_chroma_variance_gates_smoothly_rather_than_cutting_off(self) -> None:
        scores = [
            self.normalizer(
                _awb_metrics(
                    chroma_prominence=10.0,
                    chroma_periodicity=20.0,
                    chroma_variance=variance,
                    luma_chroma_decorrelation=1.0,
                )
            )
            for variance in (0.01, 0.1, 0.5, 5.0)
        ]

        self.assertEqual(scores, sorted(scores))
        self.assertEqual(len(set(scores)), len(scores))
        self.assertGreater(scores[0], 0.0, "a faint signal attenuates, it does not vanish")

    def test_decorrelation_from_luma_raises_chroma_evidence(self) -> None:
        def score(decorrelation: float) -> float:
            return self.normalizer(
                _awb_metrics(
                    chroma_prominence=10.0,
                    chroma_periodicity=20.0,
                    chroma_variance=2.0,
                    luma_chroma_decorrelation=decorrelation,
                )
            )

        self.assertLess(score(0.0), score(1.0))

    def test_takes_the_stronger_of_the_two_mechanisms(self) -> None:
        chroma_only = self.normalizer(
            _awb_metrics(chroma_prominence=10.0, chroma_periodicity=20.0, chroma_variance=2.0)
        )
        both = self.normalizer(
            _awb_metrics(
                chroma_prominence=10.0,
                chroma_periodicity=20.0,
                chroma_variance=2.0,
                ae_prominence=40.0,
                ae_periodicity=80.0,
                ae_modulation_depth=0.5,
                ae_in_band=True,
                ae_dominant_frequency=2.0,
            )
        )
        self.assertGreater(both, chroma_only)

    def test_rejects_foreign_metrics(self) -> None:
        with self.assertRaises(TypeError):
            self.normalizer(
                IlluminantMetrics(
                    dominant_frequency=0.0,
                    dominant_power=0.0,
                    peak_prominence=0.0,
                    peak_to_median_ratio=0.0,
                )
            )


class RollingBandNormalizerTests(unittest.TestCase):
    def test_still_bounded_and_rejects_foreign_metrics(self) -> None:
        normalizer = RollingBandNormalizer(
            band_strength_reference=10.0,
            velocity_reference=30.0,
            position_variance_reference=50.0,
            periodicity_reference=8.0,
        )
        saturated = normalizer(
            RollingBandMetrics(
                edge_energy=1.0,
                dominant_band_strength=1e6,
                vertical_velocity=-1e6,
                position_variance=1e6,
                temporal_periodicity=1e6,
            )
        )
        self.assertEqual(saturated, 1.0)
        with self.assertRaises(TypeError):
            normalizer(_awb_metrics())


if __name__ == "__main__":
    unittest.main()
