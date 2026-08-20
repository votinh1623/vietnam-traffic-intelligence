"""Ad hoc diagnostic: does the stillness signal separate a stalled, packed
crowd from ordinary moving traffic on a real frame pair?

Not a frozen benchmark and not wired into the pipeline: this exists to make
`src/vn_traffic/analytics/stillness.py`'s Stage 1 claim reproducible --
`bbox_union_occupancy`/ROI track count depend on detector recall, which
collapses under severe occlusion (a stalled, packed crowd), exactly when
congestion is worst. This script computes the detection-independent
stalled-dense grid for one real frame pair and writes an overlay so the
claim can be checked by eye, not just by a printed number.

Example (the case that motivated this module -- see readme.md "Detection"
and docs/benchmark_protocol.md for the pipeline run this frame is from):

    python scripts/diagnose_stillness.py \
      --source datasets/raw_videos/YTDown.com_YouTube_Rush-Hour-Traffic-with-motorcycle-in-Ho-_Media_1ZupwFOhjl4_001_1080p.mp4 \
      --frame-index 840 \
      --roi-polygon 0.48,0.05 0.62,0.05 0.84,0.95 0.30,0.95 \
      --output-dir output/stillness_diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.analytics.stillness import (  # noqa: E402
    grid_mean,
    optical_flow_magnitude,
    stalled_dense_fraction,
    stalled_dense_mask,
    texture_density,
    to_small_gray,
)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_roi_polygon(pairs: list[str] | None) -> list[tuple[float, float]] | None:
    if not pairs:
        return None
    points = []
    for pair in pairs:
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file to read")
    parser.add_argument(
        "--frame-index",
        type=int,
        required=True,
        help="Index of the second frame in the (frame-1, frame) pair to analyze",
    )
    parser.add_argument(
        "--roi-polygon",
        nargs="+",
        help="Optional normalized 'x,y' points (an existing pipeline ROI) to "
        "report separately from the rest of the frame",
    )
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--cell-px", type=int, default=8)
    parser.add_argument("--motion-threshold", type=float, default=1.0)
    parser.add_argument(
        "--texture-percentile",
        type=float,
        default=90.0,
        help="Texture threshold is set to this percentile of the frame's own "
        "texture distribution -- a fixed magic number does not transfer "
        "across videos with different compression/detail levels",
    )
    parser.add_argument(
        "--output-dir", default="output/stillness_diagnostics", type=str
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = resolve_path(args.source)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index - 1)
    ok0, frame0 = cap.read()
    ok1, frame1 = cap.read()
    cap.release()
    if not (ok0 and ok1):
        raise RuntimeError(f"failed to read frame pair ending at {args.frame_index}")

    small0 = to_small_gray(frame0, args.downscale)
    small1 = to_small_gray(frame1, args.downscale)

    motion = grid_mean(optical_flow_magnitude(small0, small1), args.cell_px)
    texture = grid_mean(texture_density(small1), args.cell_px)
    texture_threshold = float(np.percentile(texture, args.texture_percentile))
    mask = stalled_dense_mask(
        small0,
        small1,
        cell_px=args.cell_px,
        motion_threshold=args.motion_threshold,
        texture_threshold=texture_threshold,
    )

    print(
        f"grid shape: {mask.shape}, cells flagged: {int(mask.sum())}/{mask.size} "
        f"({stalled_dense_fraction(mask):.4f})"
    )
    print(
        f"motion_threshold={args.motion_threshold}, "
        f"texture_threshold={texture_threshold:.3f} (p{args.texture_percentile:.0f})"
    )

    roi_points = parse_roi_polygon(args.roi_polygon)
    overlay = frame1.copy()
    grid_h, grid_w = mask.shape
    cell_w_full, cell_h_full = width / grid_w, height / grid_h
    for gy in range(grid_h):
        for gx in range(grid_w):
            if mask[gy, gx]:
                x0, y0 = int(gx * cell_w_full), int(gy * cell_h_full)
                x1, y1 = int((gx + 1) * cell_w_full), int((gy + 1) * cell_h_full)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
    blended = cv2.addWeighted(overlay, 0.4, frame1, 0.6, 0)

    if roi_points is not None:
        roi_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
        roi_grid_pts = np.array(
            [[(round(x * grid_w), round(y * grid_h)) for x, y in roi_points]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi_mask, roi_grid_pts, 1)
        roi_mask = roi_mask.astype(bool)
        print(
            f"stalled_dense_fraction inside ROI:  "
            f"{stalled_dense_fraction(mask, roi_mask=roi_mask):.4f}"
        )
        print(
            f"stalled_dense_fraction outside ROI: "
            f"{stalled_dense_fraction(mask, roi_mask=~roi_mask):.4f}"
        )
        roi_full_pts = np.array(
            [[(round(x * width), round(y * height)) for x, y in roi_points]],
            dtype=np.int32,
        )
        cv2.polylines(blended, roi_full_pts, True, (255, 255, 0), 3)

    overlay_path = output_dir / f"frame{args.frame_index}_overlay.png"
    cv2.imwrite(str(overlay_path), blended)
    print(f"wrote {overlay_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
