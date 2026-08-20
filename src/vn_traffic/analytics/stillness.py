"""Detection-independent "stalled and visually dense" signal.

`BBoxUnionOccupancy` and the ROI track count (src/vn_traffic/analytics/
occupancy.py, engine.py) both depend on the detector successfully resolving
individual object boxes. Under severe occlusion -- a tightly packed, stopped
crowd of motorcycles -- detector recall collapses exactly when congestion is
worst, so those signals under-report occupancy precisely in the case they
are meant to catch. This was observed directly in pipeline run37 (a
rush-hour clip with a gridlocked motorcycle mass the detector drew zero
boxes over, entirely undetected and, separately, outside that run's
non-recalibrated ROI).

This module computes a coarse, per-grid-cell signal directly from pixel
motion and texture, with no dependency on any detected box, specifically to
catch that blind spot: a cell is flagged only when it is both visually dense
(high local texture -- something is there) and nearly motionless (low
optical-flow magnitude -- it is not moving). Neither signal alone is
sufficient: high texture with motion is ordinary moving traffic; low
texture with no motion is empty road.

Stage 1 only: this module computes and exposes the signal. It is not yet
wired into the congestion state machine (state.py/engine.py) -- the
thresholds a caller supplies here are illustrative, not calibrated against
any real traffic scene yet, the same honesty bar the existing congestion
thresholds are held to. It also assumes a static camera: under camera pan/
zoom (analytics.mode: uav_motion), raw optical flow reflects both camera and
object motion and would need GMC-based ego-motion compensation first, which
this module does not yet do.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_small_gray(frame_bgr: np.ndarray, downscale: int) -> np.ndarray:
    if downscale < 1:
        raise ValueError("downscale must be at least one")
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if downscale == 1:
        return gray
    height, width = gray.shape
    return cv2.resize(
        gray,
        (max(1, width // downscale), max(1, height // downscale)),
        interpolation=cv2.INTER_AREA,
    )


def optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """Dense Farneback flow magnitude, per pixel, in prev_gray's pixel units."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    return np.linalg.norm(flow, axis=2)


def texture_density(gray: np.ndarray) -> np.ndarray:
    """Local visual-detail proxy: absolute Laplacian response per pixel.

    High where the frame has fine spatial structure (a packed crowd of
    vehicles/people); low over plain, low-detail surfaces (empty asphalt,
    sky, a flat wall), regardless of whether anything there is moving.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.abs(laplacian)


def grid_mean(values: np.ndarray, cell_px: int) -> np.ndarray:
    """Box-average `values` down into cell_px x cell_px grid cells."""
    if cell_px < 1:
        raise ValueError("cell_px must be at least one")
    height, width = values.shape
    grid_width = max(1, width // cell_px)
    grid_height = max(1, height // cell_px)
    return cv2.resize(
        values.astype(np.float32),
        (grid_width, grid_height),
        interpolation=cv2.INTER_AREA,
    )


def stalled_dense_mask(
    prev_gray_small: np.ndarray,
    curr_gray_small: np.ndarray,
    *,
    cell_px: int,
    motion_threshold: float,
    texture_threshold: float,
) -> np.ndarray:
    """Per-grid-cell boolean: True where a cell is both visually dense and
    nearly motionless -- a candidate stalled/packed area, independent of any
    object detection.
    """
    motion = grid_mean(
        optical_flow_magnitude(prev_gray_small, curr_gray_small), cell_px
    )
    texture = grid_mean(texture_density(curr_gray_small), cell_px)
    return (motion < motion_threshold) & (texture > texture_threshold)


def stalled_dense_fraction(
    mask: np.ndarray, roi_mask: np.ndarray | None = None
) -> float:
    """Fraction of (optionally ROI-restricted) grid cells flagged stalled-dense."""
    if roi_mask is None:
        return float(np.mean(mask)) if mask.size else 0.0
    roi = roi_mask.astype(bool)
    if roi.shape != mask.shape or not np.any(roi):
        return 0.0
    return float(np.mean(mask[roi]))
