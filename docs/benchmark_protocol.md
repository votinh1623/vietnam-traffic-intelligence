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

Bbox-union raster-grid selection was measured over 583 acceptance frames.
Grid 1 (default) costs 3.63-5.51 ms/frame; grid 2 reduces this to
0.89-1.22 ms/frame at up to 0.007 absolute occupancy error versus grid 1;
grid 4 costs 0.26-0.29 ms/frame at up to 0.021 error. All three preserve the
two demo timelines and remain configurable for future edge benchmarks.

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

## Detector training and validation

YOLOv8s was first initialized from COCO and fine-tuned on VisDrone2019-DET as
an initialization checkpoint (`mAP50=0.389`, `mAP50-95=0.225` at the best
validation epoch, 74). The historical `vietnam_dataset_v2` run
(`mAP50=0.745`, `mAP50-95=0.481`, input 1280, `freeze=10`, optimizer `auto`)
is retained for reference only: it is **invalid for scientific comparison**
because that split leaked sources across train/validation/test.

The leakage-controlled Vietnam v5 run fine-tunes full weights (`freeze=0`)
from the VisDrone checkpoint: 30 epochs, input 1280, batch 4, AdamW at
learning rate 0.0005, weight decay 0.0005, mosaic/mixup 1.0/0.3, seed 0,
deterministic mode and AMP enabled, checkpoint selected on validation only.
Epoch 29 was selected by highest validation `mAP50-95`
(precision 0.762, recall 0.504, mAP50 0.600, mAP50-95 0.344; checkpoint
SHA-256 `729c66e676345e9c...`), with no locked-test samples used for
selection.

| Split | Precision | Recall | mAP50 | mAP50-95 | Use |
|---|---:|---:|---:|---:|---|
| Validation | 0.762 | 0.504 | 0.600 | 0.344 | Checkpoint selection |
| Locked test | 0.215 | 0.287 | 0.148 | 0.062 | Final reporting only |

The validation-to-test gap is real, not an evaluation bug: locked-test boxes
are far smaller than validation boxes (82.9%-100% of test boxes, by class,
cover under 0.1% of the image, versus 1.3%-70.0% on validation), and
pedestrian test AP50-95 is only 0.0009 while car is the strongest class at
0.193. Full hashes and per-class results are recorded in
`experiments/yolov8s_v5_locked_test_20260818/run.json`. This diagnostic does
not retune v5; v5 is a functional prototype on this locked test, not a
production-general detector.

### NWD bbox-loss ablation (rejected)

To test whether the scale-driven generalization gap above could be reduced
by a loss that stays smooth for tiny boxes, `scripts/train/nwd_loss.py`
blends CIoU with Normalized Wasserstein Distance similarity
(`alpha=0.5`, `constant=16.0` -- the constant set from the locked test's own
median box size at imgsz=1280, not the NWD paper's AI-TOD default of 12.8).
Identical dataset, initialization checkpoint, and hyperparameters as
`yolov8s_v5_seed0` (verified hash-for-hash in
`experiments/yolov8s_v5_seed0_nwd_20260819T094236/run.json`); only the bbox
loss differs. The patch itself is unit-tested against the untouched original
CIoU loss and was smoke-trained cleanly before this full run.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Locked test, baseline (CIoU) | 0.215 | 0.287 | 0.148 | 0.062 |
| Locked test, NWD (alpha=0.5, C=16) | 0.194 | 0.251 | 0.128 | 0.054 |

**This is a negative result: NWD at these settings is worse than the
baseline on every metric and every class** (per-class mAP50-95:
bus 0.036->0.019, car 0.193->0.176, motorcycle 0.068->0.057,
pedestrian 0.0009->0.0008, truck 0.014->0.015). Full record:
`benchmark_outputs/detector_v5_nwd_locked_test/run.json` (config
`configs/evaluation/detector_v5_nwd_locked_test.yaml`).

A plausible cause, not yet isolated by a further ablation: `constant=16` was
chosen from the *test* distribution, but the loss is computed on *train*
batches, where the median box is ~48px -- three times the constant. At that
ratio the NWD similarity term saturates close to zero for most training
boxes, likely supplying a weak, uninformative gradient for the majority of
training data rather than the intended smoother small-object signal. Testing
a train-distribution-scaled constant, a smaller `alpha`, or an
epoch-scheduled `alpha` (small early, larger late) would be needed before
concluding NWD cannot help here at all; none of those variants have been
run. `yolov8s_v5_seed0` (CIoU) remains the selected v5 checkpoint.

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

A controlled candidate lowered `track_high_thresh` and `new_track_thresh` to
the detector confidence of 0.4. Recall increased by 0.021, but IDF1 and MOTA
fell, precision decreased, ID switches rose by 257, and fragmentations rose by
566; it was rejected, and `bytetrack_custom.yaml` remains the selected
integration configuration
(`experiments/tracking_visdrone_mot_val_cv_v1_20260818/run.json`).

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

| Tracking profile | Frame-count macro MAE (veh/frame) | Frame-count micro WAPE | Crossing WAPE | Crossing signed error |
|---|---:|---:|---:|---:|
| Standard 640 | 10.45 | 0.372 | 0.593 | -235 |
| Standard 1280 | 9.80 | 0.319 | 0.560 | -178 |

The result is not uniformly good: 1280 frame-count WAPE reaches 0.609 on
`uav0000305_00000_v`, and both profiles severely undercount
`uav0000268_05773_v` — this demonstrates measurable counting, not
production-grade accuracy.

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

A separate, longer acceptance run (v5 checkpoint, `imgsz=640`, confidence
0.4, center-corridor ROI in `configs/pipeline/offline_video.yaml`) measured
bbox-union occupancy and speed directly:

| Acceptance clip | Evaluated span | Bbox-union occupancy median (range) | Speed median px/s | State timeline |
|---|---:|---:|---:|---|
| `traffic_jam.mp4` | 283 frames / 11.3 s | 0.589 (0.488-0.724) | 78.0 | `NORMAL` to `CONGESTED` at 2.12 s; 230 congested frames |
| `traffic_normal.mp4` | 300 frames / 10.0 s | 0.135 (0.069-0.219) | 104.6 | `NORMAL` for 300/300 frames |

Thresholds were selected only after inspecting these two timelines, so this
is deterministic separation for these two scenes and this resolution, not
evidence that thresholds generalize to another camera.

## End-to-end UAV product benchmark

`experiments/uav_pipeline_e2e_v1_20260818/run.json` records a clean-commit,
300-frame run of the complete offline CV path on a dense 1080p aerial clip.
The selected VisDrone checkpoint runs at 1280 with ByteTrack and
`max_det=1000`; analytics applies an explicit vehicle allow-list while
`tracks.csv` retains all raw model classes for audit. The measured 3.70 FPS is
end-to-end and includes inference, tracking, analytics, overlay, video writing,
and the evidence-export pass.

The artifact integrity check passed, but alert transfer did not: all 300 frames
remained `NORMAL` under a camera that zooms/pans over a visible jam. Therefore
the run is throughput and integration evidence only. It is a recorded failure
case motivating stabilization, dynamic ROI/line geometry, or BEV calibration;
its line crossings and state timeline must not be cited as physical traffic
accuracy.

## UAV moving-camera analytics (GMC)

Two fixes address the failure above, both re-run on the same real aerial clip
(`configs/pipeline/offline_video_uav_gmc.yaml`, 300 frames, VisDrone baseline
checkpoint). First, `analytics.mode: uav_motion` drops the fixed
ground-anchored ROI in favor of a full-frame region by default, and the
congestion state machine stops requiring ROI occupancy to corroborate a high
track count in this mode (`fixed_camera` keeps its original, tested
co-requirement unchanged). This alone still under-triggered: full-frame
occupancy tops out at 0.132 even in a visibly jammed scene, because a wide
aerial frame is mostly background.

Second, `analytics.gmc_enabled` adds `src/vn_traffic/analytics/motion.py`: an
ECC-based (`cv2.findTransformECC`) global motion compensator that re-projects
a hand-drawn ROI/counting-line from frame 0 into every later frame instead of
collapsing to the full frame, restoring location-specific occupancy under
pan/zoom. The transform direction is easy to get backwards without symptom;
it is verified in `tests/test_motion.py` against a known synthetic pixel
shift (the first implementation was wrong and failed that test before being
corrected). On the real clip, `gmc_consecutive_failures_at_end` was 0 across
all 300 frames (no lost lock), and the run correctly transitioned
`NORMAL`->`CONGESTED` at frame 51 with a location-specific ROI, instead of
staying `NORMAL` for all 300 frames as the original run did.

GMC is still only 2D image-plane motion compensation, not GPS/BEV
georeferencing, and can lose lock under a hard scene cut, fast motion, or
low-texture frames -- check `gmc_consecutive_failures_at_end` in
`summary.json` before trusting a given run's geometry. These runs are local
ad-hoc reruns, not a new hashed experiment record.
