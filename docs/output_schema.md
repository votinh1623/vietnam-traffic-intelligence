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
`line_crossing`, `congestion_transition`, and `prolonged_stop`; the pipeline never invents
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

A `prolonged_stop` event is emitted once when an eligible vehicle track remains
inside the ROI below the configured entry speed for the configured duration.
Its measurements contain `speed_px_s` and `stopped_duration_s`. A release-speed
hysteresis resets the alert so a later stop can emit a new event; tracking gaps
longer than `prolonged_stop_max_gap_s` reset the candidate instead of being
misread as continuous stationary evidence. This remains an image-plane motion
heuristic. Camera motion, ID switches, and perspective can invalidate a
physical-stop interpretation unless the video is stabilized/calibrated.

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
by tracker ID switches or track fragmentation. Analytics only consumes classes
listed in `analytics.included_classes`; raw `tracks.csv` may still contain
other detector classes for audit. Dense-scene label/confidence rendering can
be disabled without changing those raw rows or analytics inputs.

## `evidence.jsonl` and `evidence/`

Evidence schema version 2 links deterministic events to raw visual inputs for a
later VLM stage. Evidence extraction is an offline post-process after
`events.jsonl` is closed; it does not invoke a VLM or alter analytics results.
The exporter opens the source once and decodes frame 0 through the processed
span sequentially. It never seeks with `CAP_PROP_POS_FRAMES`, so keyframe
selection does not depend on codec/backend random-seek behavior.

```json
{
  "schema_version": 2,
  "evidence_id": "evidence-event-000004",
  "event_id": "event-000004",
  "event_type": "congestion_transition",
  "source_video_sha256": "...",
  "source_frame_index": 53,
  "source_timestamp_s": 2.12,
  "keyframe": {
    "path": "evidence/frames/event-000004.jpg",
    "frame_index": 53,
    "width": 1920,
    "height": 1080,
    "raw_bgr_sha256": "...",
    "raw_shape": [1080, 1920, 3],
    "raw_dtype": "uint8",
    "sha256": "..."
  },
  "clip": {
    "path": "evidence/clips/event-000004.mp4",
    "start_frame": 3,
    "end_frame": 128,
    "start_s": 0.12,
    "end_s": 5.12,
    "frame_count": 126,
    "fps": 25.0,
    "sha256": "..."
  }
}
```

The default policy writes a raw JPEG keyframe for `line_crossing`,
`congestion_transition`, and `prolonged_stop`, and writes pre/post clips for
congestion transitions and prolonged stops. Clip bounds are clamped to the processed video span. Files are
content-hashed, event IDs are validated before becoming filenames, and the
manifest is replaced atomically. Overlapping clip windows are written from the
same sequential decode pass. `source_video_sha256` identifies the source file;
`raw_bgr_sha256` identifies the exact decoded frame bytes before lossy JPEG
encoding, while `keyframe.sha256` and `clip.sha256` identify the exported
artifacts. Decoded-frame hashes can vary across decoder implementations, so
reproduction should also record the environment already captured by the
project. Raw evidence intentionally contains no
detection overlay; structured analytics remain available separately in the
event and timeline artifacts.

## `run.json`

The run manifest is written with `status: running` before frame processing and
atomically replaced with either `completed` or `failed`.

Stable top-level fields are:

- `schema_version`, `analytics_schema_version`, `run_id`, and `status`;
- start/completion/failure timestamps;
- resolved source, model, and config paths;
- source video properties and perception parameters;
- the evidence-selection policy and evidence export summary;
- relative artifact paths;
- processed-frame, track-row, and event counts;
- elapsed time and end-to-end processing FPS;
- an error description when the run fails.

The manifest reports pipeline throughput, not detector-only inference latency.
Stage-specific latency requires the benchmark protocol.
