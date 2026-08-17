# Offline pipeline output contract

Every successful offline-video invocation creates one immutable `runN`
directory. Analytics adds timeline and summary artifacts without changing the
meaning of the Stage 1 fields below.

## `annotated.mp4`

The source video frame rate and dimensions are preserved. Frames contain the
overlay returned by the single Ultralytics detection-and-tracking model.

## `tracks.csv`

One row represents one detection/track observation in one frame.

| Field | Type | Meaning |
|---|---|---|
| `frame_index` | integer | Zero-based decoded frame index |
| `timestamp_s` | float | `frame_index / source_fps` |
| `track_id` | nullable integer | ByteTrack identity; empty before assignment |
| `class_id` | integer | Detector class index |
| `class_name` | string | Detector class name |
| `confidence` | float | Detector confidence |
| `x1`, `y1`, `x2`, `y2` | float | Pixel-space bounding box |

## `events.jsonl`

One JSON object per deterministic analytics event. The current event types are
`line_crossing` and `congestion_transition`; the pipeline never invents
placeholder events. Analytics schema version 2 introduces union-based bbox
coverage and replaces the invalid legacy `occupancy` field. A line-crossing
event has this shape:

```json
{
  "schema_version": 2,
  "event_id": "event-000001",
  "event_type": "line_crossing",
  "timestamp_s": 0.84,
  "frame_index": 21,
  "track_id": 10,
  "class_id": 2,
  "class_name": "motorcycle",
  "direction": "down",
  "measurements": {"speed_px_s": 176.8}
}
```

A congestion transition replaces the track/class/direction fields with
`previous_state` and `current_state`; its measurements contain
`bbox_union_occupancy`, ROI-track count, and mean pixel speed. All events retain
`schema_version`, `event_id`, `event_type`, `timestamp_s`, and `frame_index`.

## `analytics.csv`

One row per processed frame, intended for timeline inspection and calibration.

| Field | Meaning |
|---|---|
| `frame_index`, `timestamp_s` | Frame position in the source video |
| `congestion_state` | `NORMAL`, `DENSE`, or `CONGESTED` after hysteresis |
| `roi_track_count` | Unique assigned track IDs currently inside the ROI |
| `bbox_union_occupancy` | Unique raster cells covered by one or more bboxes inside the ROI, divided by ROI cells |
| `mean_speed_px_s` | Mean centroid displacement rate for current ROI tracks |
| `current_counts_json` | Per-class objects currently inside the ROI |
| `cumulative_crossings_json` | Per-direction, per-class line crossings |

The ROI mask is cached and `occupancy_grid_size_px` records the raster scale;
the default value 1 uses original-resolution pixels. Bbox union avoids counting
overlap twice, but remains image-plane box coverage rather than physical road
occupancy: boxes include background and no segmentation, BEV transform, or
camera calibration is applied. Speed also remains pixel-based.
Legacy `*_occupancy` threshold keys are rejected at config load time; schema 2
requires the explicit `*_bbox_union_occupancy` names.

## `summary.json`

Run-level analytics summary containing state-frame counts, cumulative
crossings, unique track IDs, `max_bbox_union_occupancy`, maximum ROI track
count, raster grid size, and an explicit claim boundary. Counts may be biased
by tracker ID switches or track fragmentation.

## `run.json`

The run manifest is written with `status: running` before frame processing and
atomically replaced with either `completed` or `failed`.

Stable top-level fields are:

- `schema_version`, `analytics_schema_version`, `run_id`, and `status`;
- start/completion/failure timestamps;
- resolved source, model, and config paths;
- source video properties and perception parameters;
- relative artifact paths;
- processed-frame, track-row, and event counts;
- elapsed time and end-to-end processing FPS;
- an error description when the run fails.

The manifest reports pipeline throughput, not detector-only inference latency.
Stage-specific latency requires the benchmark protocol.
