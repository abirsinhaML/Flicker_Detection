"""Selection of the pixels a window's signals are measured over.

Fisheye captures do not fill their frame.  This corpus is pillarboxed: a
960x720 image circle inside a 1280x720 frame, leaving 25% of every frame as
dead black bars.  Averaging over those bars scales absolute measurements by the
crop fraction, which silently binds the calibration to one camera's geometry.

Egocentric framing adds a second dead zone.  With a field of view near 180
degrees the lower part of the frame is the wearer's own torso and hands: lit
differently from the scene, self-shadowed, and moving with the camera without
parallax.  It dilutes flicker evidence and contributes motion that is not scene
motion, so it can be excluded by policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isfinite

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegionPolicy:
    """How to choose the measured pixels of a window.

    ``border_threshold`` is a luma level below which a pixel is considered dead.
    ``exclude_bottom_fraction`` drops that share of the lowest rows.
    ``min_valid_fraction`` guards against a policy that would discard the frame.
    """

    border_threshold: float = 8.0
    exclude_bottom_fraction: float = 0.0
    min_valid_fraction: float = 0.2

    def __post_init__(self) -> None:
        if not isfinite(self.border_threshold) or self.border_threshold < 0:
            raise ValueError("border_threshold must be a finite, non-negative value")
        if not 0.0 <= self.exclude_bottom_fraction < 1.0:
            raise ValueError("exclude_bottom_fraction must be in [0, 1)")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction must be in (0, 1]")

    @classmethod
    def from_config(cls, config: dict[str, object] | None) -> RegionPolicy:
        """Build a policy from the ``analysis_region`` config block."""
        section = config or {}
        return cls(
            border_threshold=float(section.get("border_threshold", 8.0)),  # type: ignore[arg-type]
            exclude_bottom_fraction=float(
                section.get("exclude_bottom_fraction", 0.0)  # type: ignore[arg-type]
            ),
            min_valid_fraction=float(section.get("min_valid_fraction", 0.2)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class AnalysisRegion:
    """A boolean pixel mask plus the row subset worth profiling."""

    mask: np.ndarray
    rows: np.ndarray
    valid_fraction: float

    @property
    def is_full_frame(self) -> bool:
        return bool(self.mask.all())

    @classmethod
    def detect(cls, luma_frames: np.ndarray, policy: RegionPolicy) -> AnalysisRegion:
        """Find the live pixels of a ``(frames, height, width)`` luma batch.

        Deadness is judged on each pixel's *maximum* over the window, not its
        mean: a matte border stays black in every frame, whereas a genuinely
        dark corner of the scene brightens at least once.  That distinction
        keeps a dim scene from being mistaken for a crop.
        """
        if luma_frames.ndim != 3:
            raise ValueError("luma_frames must have shape (num_frames, height, width)")
        if not luma_frames.size:
            raise ValueError("luma_frames must not be empty")

        height, width = luma_frames.shape[1:]
        live = luma_frames.max(axis=0) > policy.border_threshold
        mask = cls._apply_bottom_exclusion(live, policy.exclude_bottom_fraction)

        valid_fraction = float(mask.mean())
        if valid_fraction < policy.min_valid_fraction:
            # A very dark window can look entirely dead.  Prefer measuring the
            # whole frame over reporting a signal derived from a handful of
            # pixels, and say so rather than failing quietly.
            logger.warning(
                "Analysis region kept only %.1f%% of the frame; falling back to full frame",
                valid_fraction * 100.0,
            )
            mask = cls._apply_bottom_exclusion(
                np.ones((height, width), dtype=bool),
                policy.exclude_bottom_fraction,
            )
            valid_fraction = float(mask.mean())

        return cls(
            mask=mask,
            rows=cls._profilable_rows(mask),
            valid_fraction=valid_fraction,
        )

    @classmethod
    def full_frame(cls, height: int, width: int) -> AnalysisRegion:
        """An unmasked region, used when region selection is disabled."""
        mask = np.ones((height, width), dtype=bool)
        return cls(mask=mask, rows=np.arange(height), valid_fraction=1.0)

    @staticmethod
    def _apply_bottom_exclusion(mask: np.ndarray, fraction: float) -> np.ndarray:
        if fraction <= 0.0:
            return mask
        mask = mask.copy()
        keep_rows = int(round(mask.shape[0] * (1.0 - fraction)))
        mask[max(keep_rows, 1) :, :] = False
        return mask

    @staticmethod
    def _profilable_rows(mask: np.ndarray) -> np.ndarray:
        """Rows with enough live pixels for a trustworthy row average.

        A row grazing the edge of the image circle averages over a few pixels
        and is mostly noise; including it would inject structure the rolling-band
        detector reads as banding.
        """
        counts = mask.sum(axis=1)
        if not counts.any():
            return np.arange(mask.shape[0])
        return np.flatnonzero(counts >= 0.5 * counts.max())
