"""Calibrate prolonged_stop against frame-level anomaly ground truth.

Replays the engine's windowed-drift stop logic offline over a finished
run's tracks.csv for a sweep of thresholds, and scores where it would
fire against a per-frame anomaly mask (UIT-ADrone test_frame_mask: one
0/1 label per frame, aligned 1:1 with the stitched video's frames).

Two numbers matter and they pull in opposite directions:

  precision  -- of the frames where we raise an alert, how many are
                inside a labelled anomaly. Low precision means the rule
                fires on ordinary behaviour (a vehicle legitimately
                waiting to merge is stationary too).
  segment coverage -- of the labelled anomaly segments, how many got at
                least one alert. This is the operationally meaningful
                recall: catching an incident once is what matters, not
                flagging every frame of it.

Sweeping rather than picking a threshold by intuition is the point: a
duration rule alone cannot separate "abnormally stopped" from "normally
waiting", so the sweep shows whether ANY setting separates them on real
labels, or whether the rule needs a further signal.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
import csv
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.analytics.engine import stop_drift_body_lengths  # noqa: E402
from vn_traffic.analytics.geometry import point_in_polygon, to_pixels  # noqa: E402


def load_tracks(path: Path) -> dict[int, list[dict]]:
    per_track: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row.get("track_id"):
                continue
            per_track[int(row["track_id"])].append(
                {
                    "frame_index": int(row["frame_index"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "class_name": row["class_name"],
                    "x1": float(row["x1"]), "y1": float(row["y1"]),
                    "x2": float(row["x2"]), "y2": float(row["y2"]),
                }
            )
    for rows in per_track.values():
        rows.sort(key=lambda r: r["frame_index"])
    return per_track


def fire_frames(
    per_track: dict[int, list[dict]], *, analytics: dict, roi_px,
    max_drift: float, min_duration: float, use_roi: bool,
) -> list[tuple[int, int]]:
    """Frames where prolonged_stop would fire, as (frame_index, track_id).

    Mirrors TrafficAnalytics: one event per stop episode (re-arming only
    after the track's drift exceeds the release threshold).
    """
    release = analytics.get("prolonged_stop_release_drift_body_lengths", max_drift * 2)
    max_gap = analytics["prolonged_stop_max_gap_s"]
    eligible = analytics["prolonged_stop_classes"]
    fires: list[tuple[int, int]] = []
    for track_id, rows in per_track.items():
        window: deque = deque()
        active = False
        previous_ts = None
        for row in rows:
            point = ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)
            elapsed = None if previous_ts is None else row["timestamp_s"] - previous_ts
            previous_ts = row["timestamp_s"]
            ok = (
                row["class_name"] in eligible
                and (not use_roi or point_in_polygon(point, roi_px))
                and elapsed is not None and 0 < elapsed <= max_gap
            )
            if not ok:
                window.clear()
                active = False
                continue
            window.append((row["timestamp_s"], point, max(1.0, row["y2"] - row["y1"])))
            while len(window) > 1 and row["timestamp_s"] - window[0][0] > min_duration:
                window.popleft()
            span = row["timestamp_s"] - window[0][0]
            drift = stop_drift_body_lengths(window)
            if not active and span >= min_duration and drift <= max_drift:
                active = True
                fires.append((row["frame_index"], track_id))
            elif active and drift >= release:
                active = False
    return fires


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.where(d == 1)[0].tolist(), (np.where(d == -1)[0] - 1).tolist()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--durations", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    parser.add_argument("--drifts", type=float, nargs="+", default=[0.35])
    parser.add_argument("--no-roi", action="store_true", help="ignore the ROI polygon (it is uncalibrated for new scenes)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    analytics = run["analytics"]
    width, height = run["video"]["width"], run["video"]["height"]
    roi_px = to_pixels([tuple(p) for p in analytics["roi_polygon"]], width, height)
    mask = np.load(args.mask)
    segs = segments(mask)
    per_track = load_tracks(args.run_dir / "tracks.csv")

    print(f"run={args.run_dir.name} mask={args.mask.name} frames={len(mask)} "
          f"anomaly_frames={int(mask.sum())} segments={len(segs)} "
          f"roi={'ignored' if args.no_roi else 'applied'}")
    print(f"{'drift':>6} {'min_dur':>8} {'fires':>6} {'in_anom':>8} {'precision':>10} {'segments_hit':>13}")
    for drift in args.drifts:
        for duration in args.durations:
            fires = fire_frames(
                per_track, analytics=analytics, roi_px=roi_px,
                max_drift=drift, min_duration=duration, use_roi=not args.no_roi,
            )
            frames = [f for f, _ in fires if 0 <= f < len(mask)]
            in_anomaly = sum(1 for f in frames if mask[f] == 1)
            precision = in_anomaly / len(frames) if frames else float("nan")
            hit = sum(1 for s, e in segs if any(s <= f <= e for f in frames))
            print(
                f"{drift:>6.2f} {duration:>8.1f} {len(frames):>6} {in_anomaly:>8} "
                f"{precision:>10.3f} {hit:>7}/{len(segs):<5}"
            )
    base_rate = mask.mean()
    print(f"\nbase rate (fraction of all frames labelled anomalous): {base_rate:.3f}")
    print("a precision at or below the base rate means the rule carries no information")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
