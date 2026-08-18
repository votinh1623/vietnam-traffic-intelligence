"""Rasterized bounding-box union coverage inside a configured ROI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import cv2
import numpy as np

from ..config import NormalizedPoint


Box = tuple[float, float, float, float]
PixelPoint = tuple[float, float]


def _rasterize_polygon(
    polygon_px: Sequence[PixelPoint], grid_width: int, grid_height: int
) -> np.ndarray:
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    points = np.array(
        [[(round(x), round(y)) for x, y in polygon_px]], dtype=np.int32
    )
    cv2.fillPoly(mask, points, 1)
    return mask


@dataclass(frozen=True)
class _RasterGeometry:
    roi_mask: np.ndarray
    roi_pixels: int
    grid_width: int
    grid_height: int


class BBoxUnionOccupancy:
    """Measure unique raster cells covered by any bbox inside the ROI."""

    def __init__(
        self,
        roi_polygon: tuple[NormalizedPoint, ...],
        grid_size_px: int = 1,
    ):
        if grid_size_px < 1:
            raise ValueError("grid_size_px must be at least one")
        self.roi_polygon = roi_polygon
        self.grid_size_px = grid_size_px
        self._geometry: dict[tuple[int, int], _RasterGeometry] = {}

    def _get_geometry(self, width: int, height: int) -> _RasterGeometry:
        key = (width, height)
        cached = self._geometry.get(key)
        if cached is not None:
            return cached

        grid_width = math.ceil(width / self.grid_size_px)
        grid_height = math.ceil(height / self.grid_size_px)
        roi_mask = _rasterize_polygon(
            [
                (x * width / self.grid_size_px, y * height / self.grid_size_px)
                for x, y in self.roi_polygon
            ],
            grid_width,
            grid_height,
        )
        geometry = _RasterGeometry(
            roi_mask=roi_mask,
            roi_pixels=int(np.count_nonzero(roi_mask)),
            grid_width=grid_width,
            grid_height=grid_height,
        )
        self._geometry[key] = geometry
        return geometry

    def measure(
        self,
        boxes: Iterable[Box],
        width: int,
        height: int,
        *,
        roi_polygon_px: Sequence[PixelPoint] | None = None,
    ) -> float:
        """Measure union coverage; roi_polygon_px overrides the cached static ROI.

        roi_polygon_px is already in current-frame pixel space (for example,
        a GMC-warped ROI) and is rasterized fresh every call, uncached, since
        a per-frame-varying polygon cannot reuse the static (width, height)
        cache key.
        """
        if roi_polygon_px is not None:
            grid_width = math.ceil(width / self.grid_size_px)
            grid_height = math.ceil(height / self.grid_size_px)
            scale = self.grid_size_px
            roi_mask = _rasterize_polygon(
                [(x / scale, y / scale) for x, y in roi_polygon_px],
                grid_width,
                grid_height,
            )
            roi_pixels = int(np.count_nonzero(roi_mask))
        else:
            geometry = self._get_geometry(width, height)
            roi_mask = geometry.roi_mask
            roi_pixels = geometry.roi_pixels
            grid_width = geometry.grid_width
            grid_height = geometry.grid_height
        if roi_pixels == 0:
            return 0.0

        occupied = np.zeros_like(roi_mask)
        scale = self.grid_size_px
        for x1, y1, x2, y2 in boxes:
            left = max(0, math.floor(x1 / scale))
            top = max(0, math.floor(y1 / scale))
            right = min(grid_width, math.ceil(x2 / scale))
            bottom = min(grid_height, math.ceil(y2 / scale))
            if right > left and bottom > top:
                occupied[top:bottom, left:right] = 1

        occupied_in_roi = np.count_nonzero(occupied & roi_mask)
        return float(occupied_in_roi / roi_pixels)
