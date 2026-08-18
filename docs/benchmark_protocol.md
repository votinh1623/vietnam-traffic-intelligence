# Benchmark protocol

All result rows must identify the model hash, dataset/split manifest hash,
configuration, backend, precision, seed, software environment, hardware, and
Git commit. Unexecuted measurements are recorded as `TBD`; invalidated results
remain traceable with status `invalid` and a reason.

## Runtime measurement

- use batch 1 for the real-time path and report batched throughput separately;
- perform 20–50 warm-up iterations and at least 300 timed frames;
- repeat the run three times and report sample count, mean, p50, and p95;
- synchronize CUDA around timed GPU work;
- separate cold start from warm steady-state execution;
- time input/decode, preprocess, transfer, inference, postprocess, tracking,
  analytics, evidence building, reasoning, and end-to-end latency;
- record scene density/candidate count, peak RAM/VRAM, and thermal state;
- never infer power or energy from latency.

## Quality gates

A quantized artifact is benchmarkable only after it:

1. loads successfully in the declared backend;
2. produces finite outputs with the expected shapes;
3. passes representative output parity checks;
4. is evaluated on the same locked quality set as its reference precision.

Detector quality uses mAP50, mAP50-95, per-class metrics, small-object AP, and
motorcycle AP. Tracking uses HOTA, DetA, AssA, IDF1, MOTA, ID switches, and
fragmentation. LLM/VLM quality uses a task-specific frozen evaluation set,
numeric fidelity, and unsupported-claim rate.

The v5 detector locked test was consumed once from clean commit `ac2ab2d`.
Across 176 images and 11,643 boxes it produced precision 0.215, recall 0.287,
mAP50 0.148, and mAP50-95 0.062. The complete immutable record is
`experiments/yolov8s_v5_locked_test_20260818/run.json`. These values are final
for v5 and cannot be used for further selection or tuning.

Reasoning input v1 and its no-tuning boundary are defined in
`docs/reasoning_protocol.md`. Schema validity, evidence citations, numeric
fidelity, and traffic-state fidelity are automatic gates. Human annotations
are still required before supported-claim precision, incident accuracy,
summary correctness, or quantization parity can be reported.

## Tracking evaluation status

The local motmetrics evaluator is valid for CLEAR MOT and identity metrics
after repairing its IoU-distance construction. It uses `1-IoU` directly,
converts a minimum IoU threshold `t` to the motmetrics maximum distance
`1-t`, includes the union of GT and prediction frame indices, and aggregates
sequences with a combined OVERALL accumulator.

The v5 class-aware integration baseline is recorded at
`experiments/tracking_visdrone_mot_val_v1_20260818/run.json`. It covers all
2,846 frames in seven VisDrone2019-MOT-val sequences and records `IDF1=0.309`,
`MOTA=0.020`, 462 ID switches, and MOTP distance 0.289. This result is not an
official VisDrone benchmark because non-target ignore-region handling and HOTA
are not implemented. It is also not Vietnam-domain tracking evidence.

HOTA, DetA, and AssA are not provided by motmetrics. They remain `TBD` until
TrackEval is integrated and verified on a synthetic fixture. Historical root
tracking CSV files predate the repair and remain `invalid`.

The controlled resolution experiment is recorded in
`experiments/tracking_visdrone_mot_resolution_v1_20260818/run.json`. On the
same eight vehicle classes and ByteTrack parameters, standard 1280 increased
recall from 0.449 to 0.521 and IDF1 from 0.473 to 0.481, but reduced MOTA from
0.215 to 0.132 and precision from 0.663 to 0.578. ID switches increased from
411 to 568. Resolution selection is therefore deferred to the downstream
line-crossing counting metric instead of being decided from detector AP alone.

## Small-object detection selection

The frozen comparison in
`experiments/visdrone_det_small_object_v1_20260818/run.json` evaluates four
inference modes on all 548 VisDrone2019-DET validation images and 38,759 valid
objects. It uses COCO-style AP with `maxDets=1000` and excludes VisDrone ignore
regions; it is therefore not an official VisDrone benchmark.

Standard inference at 1280 was selected. Relative to standard 640, AP increased
from 0.212 to 0.264 and AP-small from 0.118 to 0.194, with median latency rising
from 23.7 to 130.8 ms/image. SAHI 640 raised AP-small to 0.142 but reduced AP to
0.193 and raised median latency to 282.0 ms/image. Hybrid inference was worse
than the 640 reference on both AP and AP-small. No tracking or counting claim
is derived from this detector-only validation experiment.

## Counting evaluation status

The counting benchmark is frozen in
`experiments/counting_visdrone_mot_v1_20260818/run.json`. It covers 2,382
frames from six VisDrone MOT traffic sequences; the basketball-court sequence
is explicitly excluded. Standard 1280 reduced frame-count micro WAPE from
0.372 to 0.319 and line-crossing WAPE from 0.593 to 0.560 relative to the
controlled 640 profile, so it is selected for quality-first counting.

The line-crossing protocol runs the production analytics state machine at
three normalized horizontal lines. Because the UAV viewpoints move and no
stabilization or BEV transform is applied, these crossings measure agreement
with image-space GT trajectories, not calibrated physical road flow. The
remaining errors are material and must be reported: 1280 still undercounts the
aggregate frozen crossings by 178 and has line-crossing WAPE 0.560.

## Alert acceptance status

The clean acceptance record is
`experiments/alerts_acceptance_v1_20260818/run.json`. Over 180 frames,
`traffic_normal.mp4` remained `NORMAL`, while `traffic_jam.mp4` transitioned
to `CONGESTED` and remained there for 129 frames. This is a two-clip qualitative
acceptance result, not precision/recall evidence. Prolonged-stop correctness is
covered by synthetic duration, release, continuity, and tracking-gap tests;
there is no labeled real abnormal-stop clip, so real-video accuracy remains
unmeasured.
