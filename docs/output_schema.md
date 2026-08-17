# Offline pipeline output contract

Every successful offline-video invocation creates one immutable `runN`
directory. Later analytics and reasoning stages may add files, but they must
not change the meaning of the Stage 1 fields below.

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

One JSON object per deterministic analytics event. Stage 1 creates an empty
file because analytics are not implemented yet; it never invents placeholder
events. The planned common event envelope is:

```json
{
  "schema_version": 1,
  "event_id": "event-000001",
  "event_type": "congestion",
  "start_s": 42.5,
  "end_s": 58.0,
  "severity": "high",
  "measurements": {},
  "evidence": {}
}
```

Event-specific fields will be added under `measurements` and `evidence` rather
than changing the common envelope.

## `run.json`

The run manifest is written with `status: running` before frame processing and
atomically replaced with either `completed` or `failed`.

Stable top-level fields are:

- `schema_version`, `run_id`, and `status`;
- start/completion/failure timestamps;
- resolved source, model, and config paths;
- source video properties and perception parameters;
- relative artifact paths;
- processed-frame, track-row, and event counts;
- elapsed time and end-to-end processing FPS;
- an error description when the run fails.

The manifest reports pipeline throughput, not detector-only inference latency.
Stage-specific latency requires the benchmark protocol.
