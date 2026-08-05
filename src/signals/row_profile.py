"""Spatial row-profile signal extraction."""

from __future__ import annotations

import numpy as np

from src.signals.mask import AnalysisRegion

COLUMN_BANDS = 3


class RowProfileExtractor:
    """Extract per-row luminance profiles for rolling-band analysis."""

    @staticmethod
    def extract(yuv_frames: np.ndarray, region: AnalysisRegion) -> np.ndarray:
        """Return an array shaped ``(num_frames, num_profiled_rows)``.

        Each row averages only its live pixels, so a row crossing a narrow part
        of the image circle is not darkened by the dead pixels beside it.
        Without that per-row normalization the frame geometry itself appears as
        static horizontal structure, which is what banding looks like.
        """
        luma = yuv_frames[..., 0]
        mask = region.mask
        counts = np.maximum(mask.sum(axis=1), 1)
        profiles = (luma * mask).sum(axis=2) / counts
        return profiles[:, region.rows].astype(np.float32)

    @staticmethod
    def extract_column_bands(yuv_frames: np.ndarray, region: AnalysisRegion) -> np.ndarray:
        """Return row profiles per vertical slice, shaped ``(bands, frames, rows)``.

        A rolling-shutter band is imposed per sensor row and therefore spans the
        full width in phase.  Comparing slices left to right tests exactly that:
        agreement confirms straight horizontal banding, disagreement indicates the
        frame was geometrically remapped and the band model no longer applies.
        """
        luma = yuv_frames[..., 0]
        live_columns = np.flatnonzero(region.mask.any(axis=0))
        if live_columns.size < COLUMN_BANDS:
            return np.empty((0, luma.shape[0], region.rows.size), dtype=np.float32)

        bands = []
        for columns in np.array_split(live_columns, COLUMN_BANDS):
            band_mask = np.zeros_like(region.mask)
            band_mask[:, columns] = region.mask[:, columns]
            counts = np.maximum(band_mask.sum(axis=1), 1)
            profiles = (luma * band_mask).sum(axis=2) / counts
            bands.append(profiles[:, region.rows])
        return np.stack(bands).astype(np.float32)
