"""Explain why prolonged_stop did or did not fire on a completed run.

The prolonged_stop detector requires an unbroken run of frames that are
all (a) inside the ROI, (b) of an eligible class, (c) separated by no more
than max_gap_s, and (d) below max_speed_px_s -- any single frame failing
any of those resets the accumulated stop timer to zero. When the detector
reports zero events it is impossible to tell from the run summary which
of those conditions did the resetting.

This replays the same per-track computation over a finished run's
tracks.csv (no GPU, no re-inference) using the same geometry helper the
engine uses, and reports for every candidate track: the longest stop
streak it achieved, how far short of min_duration_s that fell, and which
condition ended each streak. Use it before touching any threshold, so a
change is aimed at the condition that is actually blocking.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def analyse(rows: list[dict], *, analytics: dict, roi_px) -> dict:
    """Replay the engine's stop-streak logic for one track."""
    best_streak = 0.0
    streak_start: float | None = None
    reasons = Counter()
    speeds: list[float] = []
    inside_frames = 0
    previous = None
    for row in rows:
        point = ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)
        inside = point_in_polygon(point, roi_px)
        inside_frames += int(inside)
        if previous is None:
            previous = (point, row["timestamp_s"])
            continue
        prev_point, prev_ts = previous
        elapsed = row["timestamp_s"] - prev_ts
        speed = math.dist(prev_point, point) / elapsed if elapsed > 0 else None
        if speed is not None:
            speeds.append(speed)
        previous = (point, row["timestamp_s"])

        blocker = None
        if not inside:
            blocker = "outside_roi"
        elif row["class_name"] not in analytics["prolonged_stop_classes"]:
            blocker = "class_not_eligible"
        elif speed is None or not (0 < elapsed <= analytics["prolonged_stop_max_gap_s"]):
            blocker = "gap_too_large"
        elif speed > analytics["prolonged_stop_max_speed_px_s"]:
            blocker = "speed_above_threshold"

        if blocker is None:
            if streak_start is None:
                streak_start = prev_ts
            best_streak = max(best_streak, row["timestamp_s"] - streak_start)
        else:
            if streak_start is not None:
                reasons[blocker] += 1
            streak_start = None

    return {
        "frames": len(rows),
        "class_name": rows[0]["class_name"],
        "inside_roi_frames": inside_frames,
        "median_speed": round(sorted(speeds)[len(speeds) // 2], 1) if speeds else None,
        "min_speed": round(min(speeds), 1) if speeds else None,
        "best_stop_streak_s": round(best_streak, 2),
        "streak_enders": dict(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10, help="how many closest-to-firing tracks to list")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # Thresholds come from the run's own run.json, not a config file: that
    # is exactly what the run used, and it stays readable after the config
    # is edited or removed.
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    analytics = run["analytics"]
    width = run["video"]["width"]
    height = run["video"]["height"]
    roi_px = to_pixels([tuple(p) for p in analytics["roi_polygon"]], width, height)

    per_track = load_tracks(args.run_dir / "tracks.csv")
    print(f"run={args.run_dir.name} frame={width}x{height} tracks={len(per_track)}")
    print(
        f"thresholds: max_speed={analytics['prolonged_stop_max_speed_px_s']}px/s "
        f"min_duration={analytics['prolonged_stop_min_duration_s']}s "
        f"max_gap={analytics['prolonged_stop_max_gap_s']}s "
        f"enabled={analytics['prolonged_stop_enabled']}"
    )

    results = {tid: analyse(rows, analytics=analytics, roi_px=roi_px)
               for tid, rows in per_track.items() if rows}
    eligible = {t: r for t, r in results.items() if r["class_name"] in analytics["prolonged_stop_classes"]}
    print(f"tracks of an eligible class: {len(eligible)}/{len(results)}")
    ever_inside = {t: r for t, r in eligible.items() if r["inside_roi_frames"] > 0}
    print(f"  ...that were ever inside the ROI: {len(ever_inside)}")

    all_enders = Counter()
    for r in results.values():
        all_enders.update(r["streak_enders"])
    print(f"what ended stop streaks (all tracks): {dict(all_enders)}")

    speeds = [r["min_speed"] for r in eligible.values() if r["min_speed"] is not None]
    if speeds:
        speeds.sort()
        print(
            f"per-track MINIMUM speed across eligible tracks: "
            f"min={speeds[0]:.1f} p25={speeds[len(speeds)//4]:.1f} "
            f"median={speeds[len(speeds)//2]:.1f} px/s "
            f"(threshold is {analytics['prolonged_stop_max_speed_px_s']})"
        )
        below = sum(1 for s in speeds if s <= analytics['prolonged_stop_max_speed_px_s'])
        print(f"  eligible tracks with ANY frame under the speed threshold: {below}/{len(speeds)}")

    ranked = sorted(eligible.items(), key=lambda kv: kv[1]["best_stop_streak_s"], reverse=True)
    print(f"\nclosest tracks to firing (need {analytics['prolonged_stop_min_duration_s']}s):")
    for track_id, r in ranked[: args.top]:
        print(
            f"  track {track_id:>5} {r['class_name']:<11} frames={r['frames']:>3} "
            f"in_roi={r['inside_roi_frames']:>3} best_streak={r['best_stop_streak_s']:>5.2f}s "
            f"min_speed={r['min_speed']} enders={r['streak_enders']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
