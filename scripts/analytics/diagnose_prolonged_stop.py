"""Explain why prolonged_stop did or did not fire on a completed run.

A prolonged_stop needs a track to stay inside the ROI, in an eligible
class, without a detection gap longer than max_gap_s, while its centre
wanders less than max_drift body lengths across a full min_duration_s
window. When the detector reports zero events the run summary cannot say
which of those conditions did the blocking.

This replays that computation over a finished run's tracks.csv (no GPU,
no re-inference), reusing the engine's own stop_drift_body_lengths and
point_in_polygon so the diagnosis cannot drift from the implementation.
Use it before touching any threshold, so a change is aimed at the
condition that is actually blocking.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from vn_traffic.analytics.engine import stop_drift_body_lengths  # noqa: E402
from vn_traffic.analytics.geometry import point_in_polygon, to_pixels  # noqa: E402


DEFAULT_MAX_DRIFT = 0.35


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


def analyse(rows: list[dict], *, analytics: dict, roi_px, max_drift: float) -> dict:
    """Replay the engine's windowed stop logic for one track."""
    min_duration = analytics["prolonged_stop_min_duration_s"]
    max_gap = analytics["prolonged_stop_max_gap_s"]
    eligible_classes = analytics["prolonged_stop_classes"]

    window: deque[tuple[float, tuple[float, float], float]] = deque()
    reasons: Counter[str] = Counter()
    best_span_at_rest = 0.0
    min_drift_at_full_window: float | None = None
    inside_frames = 0
    previous_ts: float | None = None

    for row in rows:
        point = ((row["x1"] + row["x2"]) / 2.0, (row["y1"] + row["y2"]) / 2.0)
        inside = point_in_polygon(point, roi_px)
        inside_frames += int(inside)
        elapsed = None if previous_ts is None else row["timestamp_s"] - previous_ts
        previous_ts = row["timestamp_s"]

        blocker = None
        if not inside:
            blocker = "outside_roi"
        elif row["class_name"] not in eligible_classes:
            blocker = "class_not_eligible"
        elif elapsed is None or not (0 < elapsed <= max_gap):
            blocker = "gap_too_large"

        if blocker is not None:
            if window:
                reasons[blocker] += 1
            window.clear()
            continue

        window.append((row["timestamp_s"], point, max(1.0, row["y2"] - row["y1"])))
        # See engine.py: trim on the second sample, not the first, or the
        # window drops below min_duration the instant it passes it.
        while len(window) > 2 and row["timestamp_s"] - window[1][0] >= min_duration:
            window.popleft()
        span = row["timestamp_s"] - window[0][0]
        drift = stop_drift_body_lengths(window)
        if drift <= max_drift:
            best_span_at_rest = max(best_span_at_rest, span)
        if span >= min_duration:
            min_drift_at_full_window = (
                drift if min_drift_at_full_window is None
                else min(min_drift_at_full_window, drift)
            )
            if drift > max_drift:
                reasons["drift_above_threshold"] += 1

    return {
        "frames": len(rows),
        "class_name": rows[0]["class_name"],
        "inside_roi_frames": inside_frames,
        "best_span_under_drift_s": round(best_span_at_rest, 2),
        "min_drift_at_full_window": (
            round(min_drift_at_full_window, 3)
            if min_drift_at_full_window is not None else None
        ),
        "blockers": dict(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    # Thresholds come from the run's own run.json -- exactly what that run
    # used, and still readable after the config is edited. Runs recorded
    # before the drift criterion replaced the px/s speeds carry no drift
    # key, so fall back to the current default for those.
    run = json.loads((args.run_dir / "run.json").read_text(encoding="utf-8"))
    analytics = run["analytics"]
    max_drift = analytics.get("prolonged_stop_max_drift_body_lengths")
    legacy = max_drift is None
    if legacy:
        max_drift = DEFAULT_MAX_DRIFT
    width, height = run["video"]["width"], run["video"]["height"]
    roi_px = to_pixels([tuple(p) for p in analytics["roi_polygon"]], width, height)

    per_track = load_tracks(args.run_dir / "tracks.csv")
    print(f"run={args.run_dir.name} frame={width}x{height} tracks={len(per_track)}")
    print(
        f"thresholds: max_drift={max_drift} body-lengths"
        f"{' (run predates drift criterion; using current default)' if legacy else ''} "
        f"min_duration={analytics['prolonged_stop_min_duration_s']}s "
        f"max_gap={analytics['prolonged_stop_max_gap_s']}s "
        f"enabled={analytics['prolonged_stop_enabled']}"
    )

    results = {
        tid: analyse(rows, analytics=analytics, roi_px=roi_px, max_drift=max_drift)
        for tid, rows in per_track.items() if rows
    }
    eligible = {
        t: r for t, r in results.items()
        if r["class_name"] in analytics["prolonged_stop_classes"]
    }
    ever_inside = sum(1 for r in eligible.values() if r["inside_roi_frames"] > 0)
    print(f"tracks of an eligible class: {len(eligible)}/{len(results)}")
    print(f"  ...ever inside the ROI: {ever_inside}")

    blockers: Counter[str] = Counter()
    for r in results.values():
        blockers.update(r["blockers"])
    print(f"what reset or blocked the stop window: {dict(blockers)}")

    min_duration = analytics["prolonged_stop_min_duration_s"]
    reached = [r for r in eligible.values() if r["min_drift_at_full_window"] is not None]
    print(f"tracks that ever held a full {min_duration}s window: {len(reached)}/{len(eligible)}")
    if reached:
        drifts = sorted(r["min_drift_at_full_window"] for r in reached)
        print(
            f"  their lowest drift over that window: min={drifts[0]:.3f} "
            f"median={drifts[len(drifts) // 2]:.3f} (threshold {max_drift})"
        )

    ranked = sorted(
        eligible.items(), key=lambda kv: kv[1]["best_span_under_drift_s"], reverse=True
    )
    print(f"\nclosest tracks to firing (need {min_duration}s under drift):")
    for track_id, r in ranked[: args.top]:
        print(
            f"  track {track_id:>5} {r['class_name']:<11} frames={r['frames']:>4} "
            f"in_roi={r['inside_roi_frames']:>4} best_span={r['best_span_under_drift_s']:>5.2f}s "
            f"min_drift={r['min_drift_at_full_window']} blockers={r['blockers']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
