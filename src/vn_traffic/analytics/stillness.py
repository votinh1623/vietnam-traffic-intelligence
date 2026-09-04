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

from typing import Sequence

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


def rasterize_roi_to_grid(
    roi_polygon_px: Sequence[tuple[float, float]],
    frame_width: int,
    frame_height: int,
    grid_height: int,
    grid_width: int,
) -> np.ndarray:
    """Rasterize a full-resolution-pixel ROI polygon at grid resolution."""
    scale_x = grid_width / frame_width
    scale_y = grid_height / frame_height
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    points = np.array(
        [[(round(x * scale_x), round(y * scale_y)) for x, y in roi_polygon_px]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, points, 1)
    return mask.astype(bool)


class StillnessTracker:
    """Stateful per-frame wrapper: tracks the previous downscaled grayscale
    frame and reports the stalled-dense fraction each call, restricted to a
    caller-supplied ROI when given. Stage 2 integration point for
    `CongestionStateMachine` -- see its `stalled_dense_fraction` parameter.

    Thresholds are absolute and demo-calibrated on one real frame pair, the
    same honesty bar the existing congestion thresholds are held to, not
    tuned across multiple scenes.
    """

    def __init__(
        self,
        *,
        downscale: int = 4,
        cell_px: int = 8,
        motion_threshold: float = 1.0,
        texture_threshold: float = 250.0,
    ):
        self.downscale = downscale
        self.cell_px = cell_px
        self.motion_threshold = motion_threshold
        self.texture_threshold = texture_threshold
        self._previous_small_gray: np.ndarray | None = None

    def update(
        self,
        frame_bgr: np.ndarray,
        *,
        roi_polygon_px: Sequence[tuple[float, float]] | None = None,
    ) -> float:
        """Feed the next frame in sequence; returns 0.0 on the first call
        (no previous frame to compare motion against) and thereafter the
        stalled-dense fraction, restricted to `roi_polygon_px` if given.
        """
        height, width = frame_bgr.shape[:2]
        small = to_small_gray(frame_bgr, self.downscale)
        if self._previous_small_gray is None:
            self._previous_small_gray = small
            return 0.0

        mask = stalled_dense_mask(
            self._previous_small_gray,
            small,
            cell_px=self.cell_px,
            motion_threshold=self.motion_threshold,
            texture_threshold=self.texture_threshold,
        )
        self._previous_small_gray = small

        if roi_polygon_px is None:
            return stalled_dense_fraction(mask)
        grid_height, grid_width = mask.shape
        roi_mask = rasterize_roi_to_grid(
            roi_polygon_px, width, height, grid_height, grid_width
        )
        return stalled_dense_fraction(mask, roi_mask=roi_mask)


def stalled_dense_score(
    prev_gray_small: np.ndarray,
    curr_gray_small: np.ndarray,
    *,
    cell_px: int,
    motion_threshold: float,
    texture_percentile: float = 90.0,
) -> np.ndarray:
    """Per-cell continuous "stalled and dense" concern score in [0, 1].

    Unlike `stalled_dense_mask` (a fixed absolute texture threshold, meant
    to be comparable across frames for a state-machine trigger -- see
    `StillnessTracker`), this uses a frame-RELATIVE texture threshold (this
    frame's own `texture_percentile`). A relative threshold cannot produce a
    scalar that is comparable across frames or scenes (by construction it
    always flags roughly the same top fraction of any frame, regardless of
    how severe the scene actually is -- confirmed empirically: a fixed
    absolute threshold's fraction stayed flat at ~0.15-0.20 across an entire
    real 900-frame clip whether the visible scene was light or gridlocked).
    It IS however validated to spatially localize a real packed/stalled
    cluster well, frame by frame, across a real clip -- see
    `StillnessHeatmapRenderer`'s heatmap mode
    by enabling `stillness_heatmap.enabled` in a local pipeline config. Intended
    for a visual heatmap a human operator reads, not as a state-machine
    trigger.
    """
    motion = grid_mean(optical_flow_magnitude(prev_gray_small, curr_gray_small), cell_px)
    texture = grid_mean(texture_density(curr_gray_small), cell_px)
    texture_threshold = float(np.percentile(texture, texture_percentile))
    still = motion < motion_threshold
    denom = max(float(texture.max() - texture_threshold), 1e-6)
    normalized_texture = np.clip((texture - texture_threshold) / denom, 0.0, 1.0)
    return np.where(still, normalized_texture, 0.0).astype(np.float32)


WATERMARK_TEXT = "RELATIVE STILLNESS-TEXTURE (not a congestion decision)"


def render_heatmap_overlay(
    frame_bgr: np.ndarray,
    score_grid: np.ndarray,
    *,
    alpha_max: float = 0.5,
    watermark: bool = True,
) -> np.ndarray:
    """Blend `score_grid` onto `frame_bgr` as a JET-colormap heatmap.

    Cells with score 0 are left untouched (alpha 0) so the original frame
    shows through everywhere the signal did not fire, not just faded. A
    still building facade, signage, or standing crowd can score just as
    high as a stalled vehicle cluster (texture alone has no notion of
    "vehicle"; see stalled_dense_score's docstring) -- this is a relative
    stillness/texture map, not a congestion probability, which is why a
    literal watermark is burned into the frame by default rather than left
    to documentation alone.
    """
    height, width = frame_bgr.shape[:2]
    heat_u8 = np.clip(score_grid * 255.0, 0, 255).astype(np.uint8)
    heat_full = cv2.resize(heat_u8, (width, height), interpolation=cv2.INTER_NEAREST)
    heat_color = cv2.applyColorMap(heat_full, cv2.COLORMAP_JET)
    alpha = (heat_full.astype(np.float32) / 255.0 * alpha_max)[..., None]
    blended = frame_bgr.astype(np.float32) * (1 - alpha) + heat_color.astype(
        np.float32
    ) * alpha
    result = blended.astype(np.uint8)
    if watermark:
        y = height - 12
        cv2.putText(
            result, WATERMARK_TEXT, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            result, WATERMARK_TEXT, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return result


class StillnessHeatmapRenderer:
    """Stateful per-frame wrapper: computes `stalled_dense_score` from
    consecutive raw frames and blends it onto a (possibly already
    detection/analytics-annotated) display frame. Purely a visualization
    aid -- independent of `TrafficAnalytics`/`CongestionStateMachine`, does
    not affect any deterministic analytics output or state decision. See
    `stalled_dense_score` for why this uses a frame-relative threshold
    instead of `StillnessTracker`'s fixed one.

    Applies decay-from-peak persistence across frames (`smoothing_decay`)
    before rendering: `persistent = max(current_raw, persistent_prev *
    smoothing_decay)`. The raw per-frame score flickers -- measured on 10
    consecutive real frames of the motivating jam clip, the thresholded
    mask's frame-to-frame IoU averaged only 0.653 (min 0.519), roughly a
    third of the flagged cells changing identity every frame, which reads
    as noise rather than a confident highlight when watched as video, even
    though a single still frame looks fine. An exponential-moving-AVERAGE
    was tried first and rejected: it raised IoU further (0.862) but also
    washed out peak brightness for any cell not flagged in every single
    recent frame, which in a real pipeline run made the heatmap barely
    visible over the actual jam despite the IoU number looking good. Decay-
    from-peak (a MAX, not a weighted average) keeps peak brightness at 1.0
    on any currently- or recently-flagged cell while still raising IoU to
    0.789 (min 0.634) at `smoothing_decay=0.95` on the same frames.
    """

    def __init__(
        self,
        *,
        downscale: int = 4,
        cell_px: int = 8,
        motion_threshold: float = 1.0,
        texture_percentile: float = 85.0,
        alpha_max: float = 0.5,
        smoothing_decay: float = 0.95,
    ):
        self.downscale = downscale
        self.cell_px = cell_px
        self.motion_threshold = motion_threshold
        self.texture_percentile = texture_percentile
        self.alpha_max = alpha_max
        self.smoothing_decay = smoothing_decay
        self._previous_small_gray: np.ndarray | None = None
        self._persistent_score: np.ndarray | None = None

    def render(self, raw_frame_bgr: np.ndarray, display_frame_bgr: np.ndarray) -> np.ndarray:
        """`raw_frame_bgr` drives the motion/texture computation (must be the
        unannotated source frame); the resulting heatmap is blended onto
        `display_frame_bgr` (which may already carry detection boxes/other
        overlays) and returned. Returns `display_frame_bgr` unchanged on the
        first call (no previous frame yet).
        """
        small = to_small_gray(raw_frame_bgr, self.downscale)
        if self._previous_small_gray is None:
            self._previous_small_gray = small
            return display_frame_bgr
        score = stalled_dense_score(
            self._previous_small_gray,
            small,
            cell_px=self.cell_px,
            motion_threshold=self.motion_threshold,
            texture_percentile=self.texture_percentile,
        )
        self._previous_small_gray = small
        if self._persistent_score is None or self._persistent_score.shape != score.shape:
            self._persistent_score = score
        else:
            self._persistent_score = np.maximum(
                score, self._persistent_score * self.smoothing_decay
            )
        return render_heatmap_overlay(
            display_frame_bgr, self._persistent_score, alpha_max=self.alpha_max
        )
