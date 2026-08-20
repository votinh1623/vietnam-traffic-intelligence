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

### P2 detection-head ablation (rejected)

To test whether the same scale-driven generalization gap could be reduced
from the architecture side instead of the loss side, `configs/experiments/
architectures/yolov8s-p2-vietnam.yaml` adds a 4th Detect scale at stride 4
(P2) to the plain 3-head model (stride 8/16/32), whose smallest stride gives
no output a receptive field sized for objects a few pixels wide -- most of
the locked-test box-size distribution. Backbone weights transfer from the
same VisDrone checkpoint as baseline/NWD (`YOLO(architecture_yaml).load(pt)`,
219/437 state-dict items transfer -- the rest is the architecturally new
head, which initializes fresh).

**Training incident history (must be disclosed alongside any P2 result).**
The first training attempt (`yolov8s_v5_seed0_p2`, batch=2 -- batch=4
measured 5.99/6.00 GiB reserved for a single forward+backward alone on this
GPU, so batch was halved) trained cleanly through epoch 17 (best.pt at
epoch 15, mAP50-95=0.28423, independently re-verified via a standalone
`model.val()` call). It then crashed silently (process killed externally,
no catchable exception) partway through a `resume=True` continuation.
Recovery took three attempts: a real `RuntimeError: CUDA error: out of
memory` (allocator fragmentation, not a sizing error); a batch=1 retry that
avoided the OOM but corrupted BatchNorm statistics (precision 0.810, recall
0.020, mAP50-95 0.001 -- BatchNorm needs batch ≥ 2 for meaningful running
statistics); and a batch=2 `resume=True` retry that silently re-transferred
only 219/437 items from the *original external VisDrone checkpoint* instead
of continuing from this run's own state (`mAP50-95` collapsed to 0.002) --
an Ultralytics `resume=True` bug interacting with this project's
`architecture_yaml` + `.load()` checkpoint-construction pattern. All three
failed attempts polluted `yolov8s_v5_seed0_p2`'s `results.csv` (permanently,
Ultralytics only appends) and `last.pt` (unusable); `best.pt` (epoch 15) was
never touched by any of them. The original run's manifest is marked
`aborted_invalid_provenance`:
`experiments/yolov8s_v5_seed0_p2_20260819T114628/run.json`.

Recovery abandoned `resume=True` entirely: `yolov8s_v5_seed0_p2_continued`
is a plain, non-resume `YOLO(best.pt).train(...)` call for 15 more epochs in
a new output directory, using the verified-good epoch-15 `best.pt` as an
ordinary pretrained checkpoint. This means optimizer state, LR schedule, and
warmup all reinitialize from scratch -- it is a **two-stage/restarted
fine-tune**, not epochs 18-30 of one continuous training curriculum, and
must be reported as such. It also inherits the batch=2 confound (not held
constant against baseline/NWD's batch=4; Ultralytics' `nbs=64` gradient
accumulation partially compensates but this is undocumented elsewhere).
Manifest: `experiments/yolov8s_v5_seed0_p2_continued_20260819T152917/run.json`.

**Results.** Best checkpoint by validation `mAP50-95` was epoch 13 of the
continuation (precision 0.652, recall 0.500, mAP50 0.552, mAP50-95 0.321);
epochs 9-15 plateau in the 0.310-0.321 range with no further upward trend.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Validation, baseline (CIoU) | 0.762 | 0.504 | 0.600 | 0.344 |
| Validation, NWD | 0.711 | 0.529 | 0.600 | 0.337 |
| Validation, P2 (epoch 13) | 0.652 | 0.500 | 0.552 | 0.321 |
| Locked test, baseline (CIoU) | 0.215 | 0.287 | 0.148 | 0.062 |
| Locked test, NWD | 0.194 | 0.251 | 0.128 | 0.054 |
| Locked test, P2 | 0.146 | 0.206 | 0.095 | 0.037 |

**This is a negative result: P2 is worse than both baseline and NWD on
every metric, on both splits** (per-class locked-test mAP50-95: bus
0.036->0.003, car 0.193->0.130, motorcycle 0.068->0.041, pedestrian
0.0009->0.0008 -- only 170 pedestrian boxes in locked test, treat with
caution -- truck 0.014->0.008). Its validation-to-test drop (0.321->0.037,
~8.8x) is also proportionally larger than baseline's (~5.5x) and NWD's
(~6.2x), despite directly targeting that gap. Full record:
`benchmark_outputs/detector_v5_p2_locked_test/run.json` (config
`configs/evaluation/detector_v5_p2_locked_test.yaml`).

Two confounds prevent a clean causal read of "P2 architecture is worse":
the batch=2 vs batch=4 gap, and the two-stage/restarted fine-tune (a fresh
optimizer/LR-warmup restart partway through what would ideally be one
continuous curriculum). Both plausibly cost some accuracy independent of
the architecture change itself. Additionally, like NWD's constant, P2's
design (adding a scale specifically sized for the locked test's median
16px box) was chosen with knowledge of the locked-test box-size
distribution -- this result is exploratory, not a blind confirmatory test.
`yolov8s_v5_seed0` (CIoU, 3-head) remains the selected v5 checkpoint. With
both a loss-side fix (NWD) and an architecture-side fix (P2) now rejected,
the next candidate for this gap is copy-paste augmentation of small
objects, which addresses the underlying training-data scarcity directly
instead of asking the loss or architecture to compensate for it.

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

HOTA, DetA, and AssA are not provided by motmetrics; they are computed
separately via `scripts/evaluate_hota.py` (TrackEval's own implementation,
verified on synthetic fixtures in `tests/test_hota_metrics.py`) and
reported in the readme's Tracking section. That script originally
pre-filtered ground truth to prediction-only frames, silently dropping any
frame with zero predicted boxes instead of scoring it as a false negative;
fixed and re-run (moved every metric by at most 0.002 -- see the readme's
Tracking caveat). Historical root tracking CSV files predate the motmetrics
repair and remain `invalid`.

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

**This section previously overclaimed GMC's effect and is corrected here
with a real A/B test.** `analytics.mode: uav_motion` drops the fixed
ground-anchored ROI in favor of a full-frame region by default, and the
congestion state machine stops requiring ROI occupancy to corroborate a high
track count in this mode (`fixed_camera` keeps its original, tested
co-requirement unchanged -- and per the confidence/count experiment above,
that co-requirement's absence is exactly what made `uav_motion`'s
count-alone trigger unsafe on `traffic_normal.mp4`). `analytics.gmc_enabled`
additionally adds `src/vn_traffic/analytics/motion.py`: an ECC-based
(`cv2.findTransformECC`) global motion compensator that re-projects a
hand-drawn ROI/counting-line from frame 0 into every later frame instead of
collapsing to the full frame. The transform direction is easy to get
backwards without symptom; it is verified in `tests/test_motion.py` against
a known synthetic pixel shift (the first implementation was wrong and
failed that test before being corrected).

The readme previously claimed GMC (not just `uav_motion` mode) "fixed"
congestion detection on this clip. That claim was never isolated from
`uav_motion` mode's own count-alone trigger, and a direct A/B check shows
it does not hold: re-running the same clip, model, ROI, and thresholds with
only `gmc_enabled` toggled --

| | GMC off (`run50`) | GMC on (`run51`) |
|---|---:|---:|
| Transition frame | 64 (t=2.14s) | 65 (t=2.17s) |
| State | NORMAL 64 / CONGESTED 236 | NORMAL 65 / CONGESTED 235 |
| Max occupancy | 0.174 | 0.126 |
| Max ROI count | 171 | 130 |

-- both reach `CONGESTED` at effectively the same frame (1 frame / 33ms
apart, within noise). **`uav_motion` mode's count-alone trigger causes the
transition regardless of GMC**; GMC changes the occupancy/count magnitudes
(a more accurate, location-specific ROI under pan/zoom, not diluted the
same way a static full-frame region is) but is not what flips the state
here. GMC's real, still-untested value proposition is ROI/counting-line
*positional accuracy* under pan/zoom, not congestion-state triggering --
that would need a positional-accuracy check (e.g. does the re-projected ROI
actually track a fixed real-world region), not a state-transition
comparison, and has not been done.

The original "0 GMC lock failures across all 300 frames" claim was also an
artifact of only reporting `gmc_consecutive_failures_at_end` (the failure
streak still active at the very last frame). `GlobalMotionCompensator` now
also tracks `total_failures` (never resets, unlike `consecutive_failures`),
exposed as `gmc_total_failures` in `summary.json`
(`tests/test_motion.py::test_total_failures_does_not_reset_on_recovery`
covers the resets vs. accumulates distinction). Re-running with this field
available: `run51` (GMC on) shows `gmc_total_failures=1` despite
`gmc_consecutive_failures_at_end=0` -- the run did lose lock once and
recover, which the old single-field reporting would have hidden entirely.

GMC is still only 2D image-plane motion compensation, not GPS/BEV
georeferencing, and can lose lock under a hard scene cut, fast motion, or
low-texture frames -- check `gmc_total_failures` (run-wide) and
`gmc_consecutive_failures_at_end` (end-of-run streak only) in
`summary.json` before trusting a given run's geometry. `run50` and `run51`
are local ad-hoc reruns (`configs/pipeline/offline_video_uav_gmc.yaml` and
its `offline_video_uav_gmc_off_ab.yaml` sibling), not a new hashed
experiment record.

## Detection-independent stillness signal (prototype)

**Motivation.** A local pipeline run (fixed-camera profile, 900 frames of a
real Ho Chi Minh City rush-hour clip) reached `DENSE` but never `CONGESTED`,
even though the source clearly contains a gridlocked motorcycle mass. Two
compounding causes were confirmed by inspecting the run's own
`latest_frame.jpg`: the hand-drawn ROI (calibrated for a different clip, not
this one) does not cover most of that mass, and -- separately, visible in
the same frame -- the detector draws **zero boxes** over the packed cluster
itself, while individually-spaced vehicles elsewhere in the same frame are
detected normally. `bbox_union_occupancy` and ROI track count are both
detection-dependent, so detector recall collapsing under severe occlusion
produces a structural blind spot exactly when congestion is worst: the more
severely jammed a scene gets, the fewer boxes the detector returns, so the
measured occupancy goes *down*, not up, in the extreme regime.

**What was built (Stage 1).** `src/vn_traffic/analytics/stillness.py`
computes a coarse, per-grid-cell signal directly from pixel motion (dense
Farneback optical flow magnitude) and local texture (absolute Laplacian
response), with no dependency on any detected box. A cell is flagged
"stalled-dense" only when it is both visually dense (something is there)
and nearly motionless (it is not moving) -- neither signal alone
distinguishes a stalled crowd from an empty road (no texture, no motion) or
ordinary flowing traffic (texture, but with motion). Covered by 6 unit
tests (`tests/test_stillness.py`) with synthetic frames: a static textured
frame is flagged, a static flat (textureless) frame is not, a moving
textured frame is not, and grid reduction/ROI-restriction behave as
specified.

**Qualitative validation against the real failure case.**
`scripts/diagnose_stillness.py` reproduces this on the actual frame pair
(frame 839->840 of the rush-hour clip above) that motivated the module:

```
python scripts/diagnose_stillness.py \
  --source datasets/raw_videos/YTDown.com_YouTube_Rush-Hour-Traffic-with-motorcycle-in-Ho-_Media_1ZupwFOhjl4_001_1080p.mp4 \
  --frame-index 840 \
  --roi-polygon 0.48,0.05 0.62,0.05 0.84,0.95 0.30,0.95
```

With the texture threshold set to that frame's own 90th percentile (a fixed
magic number would not transfer across videos with different compression or
detail levels) and `motion_threshold=1.0`, 10.0% of the frame's grid cells
are flagged (198/1980), and the overlay shows them concentrated almost
exactly over the packed motorcycle/pedestrian mass -- not over empty road or
the individually-tracked moving vehicles. Mean optical-flow magnitude inside
the run's ROI (the flowing lane) is 2.218 versus 0.479 outside it (where
most of the jam sits), consistent with the jam being genuinely stalled, not
merely undetected.

**Caveats (why this is Stage 1 only, not a shipped feature).** Not wired
into `state.py`/`engine.py`: thresholds here are illustrative and
frame-relative (a percentile of that one frame's own texture distribution),
not calibrated against multiple scenes the way the existing congestion
thresholds are (themselves only two-video demo calibration, per Known
limitations). High-contrast static surfaces unrelated to traffic (building
signage, in the same overlay) also flag as false positives, since texture
alone cannot distinguish "packed vehicles" from "any static detailed
surface" -- a real, disclosed limitation, not hidden by this validation.
The module assumes a static camera: under `analytics.mode: uav_motion`, raw
optical flow reflects both camera and object motion and would need
GMC-based ego-motion compensation first, which it does not yet do. This is
one real frame pair, qualitatively checked by eye against a visual overlay,
not a multi-scene, hash-pinned benchmark.

**Stage 2 (wired, first real-pipeline result -- threshold does not yet
transfer).** `analytics.stillness_enabled` adds `StillnessTracker` to
`TrafficAnalytics` (`engine.py`) and a `stalled_dense_fraction` term to
`CongestionStateMachine._target()` (`state.py`), corroborating `CONGESTED`
independent of `bbox_union_occupancy`/count -- and deliberately **not**
gated by the detected mean speed, since that speed is exactly the signal a
severely occluded jam starves. Covered by 5 new unit/integration tests
(`tests/test_traffic_analytics.py`): the state machine reaches `CONGESTED`
from a low-occupancy, low-count, high-stillness input; a `CONGESTED` state
held by stillness ignores a high *detected* speed on release; the signal is
a no-op when `stillness_enabled=False`; and a full `TrafficAnalytics` run
with **zero detected tracks** reaches `CONGESTED` from static, textured
synthetic frames alone. `analytics.csv`/`AnalyticsSnapshot` gained a
`stalled_dense_fraction` column (`ANALYTICS_SCHEMA_VERSION` 2 -> 3).

Per the project's rule to verify any pipeline-runtime change with a real
run, not just unit tests, `configs/pipeline/offline_video_stillness_demo.yaml`
re-runs the exact motivating clip (900 frames,
`YTDown.com_..._Rush-Hour-Traffic-with-motorcycle-in-Ho-...`) with
`stillness_enabled=true` and a full-frame ROI (unlike
`offline_video.yaml`'s hand-drawn trapezoid, which does not cover most of
this clip's jam). Result: `stalled_dense_fraction` peaked at **0.214**
across all 900 frames -- below the Stage 1 demo threshold
(`stillness_congested_enter_fraction=0.30`) -- so the run stayed at `DENSE`
(846 `NORMAL` / 54 `DENSE` frames), never reaching `CONGESTED` via
stillness. This is not a bug: the wiring computed and compared real numbers
correctly end to end (`output/pipeline/run38`); the *specific threshold*,
calibrated from one frame's 90th-percentile texture value, undershoots on
this full clip once averaged over 900 frames and diluted by a full-frame
ROI (buildings/sky/road count equally alongside the jam). This is the same
lesson as the NWD ablation's mis-scaled constant: a threshold picked from a
single data point does not transfer, and is exactly why Stage 2 was
disclosed as "not calibrated across multiple scenes" rather than shipped as
final. The threshold was deliberately left as measured here, not tuned
downward after the fact to force a passing demo.

**Two follow-up hypotheses tested and rejected, before finding what
actually works.** After the flat-fraction finding above, two explanations
were tested directly on real data before concluding the fixed-threshold
scalar approach itself was the wrong tool:

1. *Background contamination (buildings always static+textured, diluting
   the fraction)* -- tested by building a per-cell "ever showed motion in
   this 900-frame clip" activity mask. Rejected: the packed motorcycle mass
   itself never moved during this whole clip (a persistent gridlock, not
   one that forms partway through), so it looks identical to a building
   under an activity-history test -- this mask would exclude the jam
   itself, not just buildings.
2. *ROI too broad (full-frame dilutes both occupancy and stillness alike)*
   -- tested by restricting the fixed-threshold fraction to a road-only ROI
   (excluding the top ~25% building strip). Rejected: `mean=0.142,
   std=0.010` versus full-frame's `mean=0.170, std=0.016` -- removing
   buildings did not reveal a hidden discriminative signal; the fraction
   stayed just as flat.

**Root cause, confirmed.** The fixed absolute `texture_threshold` cannot
discriminate severity because Laplacian-based "texture" has no notion of
*vehicle-ness* -- a packed motorcycle mass and a building facade both
register as "high spatial-frequency content," so neither a better ROI nor a
motion-history filter can separate them; the feature choice itself was the
limitation, not the threshold value or the region it was applied to.

**What actually works: a per-frame-relative heatmap, not a cross-frame
scalar.** The single-frame Stage 1 diagnostic (a `texture_percentile` of
*that frame's own* distribution, not a fixed absolute number) had already
spatially localized the jam correctly -- the flat-fraction problem only
appeared when that was converted into a fixed absolute threshold for a
comparable-over-time scalar. `stalled_dense_score()` restores the
frame-relative approach and returns a continuous per-cell score instead of
a hard mask, rendered as a heatmap
(`render_heatmap_overlay`/`StillnessHeatmapRenderer`) -- a visualization
for a human operator, deliberately not fed back into
`CongestionStateMachine`. Checked on four real frames spread across the
900-frame clip (100, 400, 700, 840): the tinted region consistently
concentrates on the packed motorcycle/pedestrian mass in every frame, not
on buildings or individually-moving vehicles.

Wired as a fully independent pipeline layer (`stillness_heatmap.*` in
`configs/pipeline/*.yaml`, `StillnessHeatmapConfig`, `PipelineRunner`'s new
optional `heatmap_renderer`) so it does not require
`analytics.stillness_enabled`/`enabled` at all. Verified with a real
pipeline run per project convention:
`configs/pipeline/offline_video_stillness_heatmap_demo.yaml`, 300 frames of
the same motivating clip (`output/pipeline/run39`) -- the tint visibly
covers the same gridlocked motorcycle mass the detector draws zero boxes
over, while individually-detected vehicles on open road stay untinted.
9 unit tests (`tests/test_stillness.py`, `tests/test_pipeline_config.py`)
cover the score function, the blend/no-op behavior, and config
load/validation.

**Checked on a confirmed non-congested clip with visible static
background.** `configs/pipeline/offline_video_stillness_heatmap_nojam_check.yaml`
re-runs the Hanoi rush-hour clip (confirmed `NORMAL` 300/300, see the
confidence/count experiment below) with the heatmap enabled
(`output/pipeline/run49`). No dramatic false-tint of plain building
facades was observed; there is faint, intermittent tinting over a cluster
of motorcycles parked at a roadside market -- consistent with, not
contrary to, the disclosed limitation that this signal responds to
"static and visually dense," not specifically "traffic jam" (parked
motorcycles are genuinely static and detailed, just not a road
congestion event). This is a qualitative check on one clip, not a
false-positive rate.

**Labeling.** This is a *relative stillness-texture overlay*, not a
congestion heatmap: `render_heatmap_overlay` now burns a literal watermark
("RELATIVE STILLNESS-TEXTURE (not a congestion decision)") into every
frame it renders, on by default, so the visualization cannot be shipped or
screenshotted without that label attached. It is not, and must not become,
an input to `CongestionStateMachine` or a fact asserted to the VLM/LLM
stage -- `analytics.stillness_enabled` (the state-machine trigger) and
`stillness_heatmap.enabled` (this overlay) are separate config flags for
exactly that reason, and `analytics.stillness_enabled` is now rejected at
config-load time when `analytics.mode: uav_motion` (raw optical flow is
invalid under camera pan/zoom; see `validate_analytics_config` in
`src/vn_traffic/config.py`).

**Next steps, explicitly staged.** The scalar `CongestionStateMachine`
trigger (Stage 2) is now root-caused, not just "needs more calibration
data": a fixed-threshold Laplacian scalar structurally cannot discriminate
severity, so recalibrating its threshold across more scenes would not fix
it -- it would need a different feature entirely (see "next candidate
features" below), which is a bigger, not-yet-scoped investment, not a
tuning pass. The heatmap (Stage 2b) is done and real-pipeline-validated as
a visualization aid, decoupled from that scalar. Stage 3 (not started):
decompose a single ROI into multiple named sub-regions (lanes) so "one lane
jammed, one lane flowing" produces two distinct states instead of one
diluted aggregate number -- worth revisiting once a working per-cell
severity feature exists, since region decomposition alone does not fix the
underlying feature-choice problem either. Next candidate features for a
real automatic trigger, neither implemented nor validated yet: low-confidence
pre-NMS detector proposals as a coarse density prior (reintroduces partial
detector dependence, but tolerant of occlusion since it does not need
resolved instances), or a texture filter band-passed to the vehicle/head
size at this camera's resolution instead of raw Laplacian's scale-agnostic
response.

### A partial, real fix found: lower detector confidence, keep occupancy corroboration

Before building a separate low-confidence inference pass, two cheap,
config-only tests (no code changes) were tried directly on the real
pipeline, in decreasing order of how promising they first looked and how
much they actually held up:

**Test 1 -- lower `perception.confidence` alone (0.4 -> 0.1).** Detector
recall under severe occlusion is gated partly by the confidence threshold;
lowering it lets more partial/low-confidence detections through. On the
motivating rush-hour clip (`fixed_camera` mode, full-frame ROI, 900
frames): `DENSE` frames rose from 54 to 243 (6% to 27%), `roi_track_count`
now sits at 50-92 across nearly the whole clip (was much lower before), but
`bbox_union_occupancy` still only reaches 0.28-0.33 (below the 0.50
`CONGESTED` entry threshold) because the severely occluded core of the
crowd still contributes far fewer boxes than its true density. **A real,
disclosed, partial improvement** -- `CONGESTED` is still never reached, and
73% of the clip stays `NORMAL` despite the crowd being visibly packed from
frame 0 (confirmed persistent, not building up -- see the activity-mask
result above).

**Test 2 -- also drop the occupancy co-requirement (`analytics.mode:
uav_motion`, which lets a high `roi_track_count` trigger `CONGESTED`
without corroborating occupancy).** With confidence still at 0.1, this
looked dramatic at first: `CONGESTED` for 839/900 frames (93%), a single
clean transition, matching human perception of the clip far better than
Test 1. **Rejected after a false-positive check on `traffic_normal.mp4`,
a genuinely light-traffic reference clip**: the same config gave
`CONGESTED` for 238/300 frames (79%) there too.
`roi_track_count` reached 100-122 on the *normal* clip -- comparable to or
higher than the jam clip's 64-94 -- while `bbox_union_occupancy` correctly
stayed low (0.06-0.11 vs the jam clip's 0.15-0.32). **Root cause:** the
`uav_motion` mode's occupancy co-requirement was assumed (based on its
existing code comment) to exist only to denoise a small, easily-saturated
ROI, and therefore safe to drop for a full-frame ROI. That assumption was
wrong: occupancy corroboration is what separates "many vehicles because the
view is wide" from "many vehicles because they are packed and stalled" --
count alone does not make that distinction on real data, regardless of ROI
size. Bypassing it reintroduces exactly the false-positive risk it exists
to prevent.

**No false `CONGESTED`/`DENSE` transition was observed** on 4 additional
real clips chosen as known-non-congested by content/title, `fixed_camera`
mode with confidence=0.1 (Test 1's config): `traffic_normal.mp4` (NORMAL
300/300), `vid3.MP4` (NORMAL 300/300, light traffic), a Hanoi rush-hour
clip (NORMAL 300/300, busy but flowing -- max occupancy 0.25, just under
the 0.30 `DENSE` entry threshold), and `DJI_20250516071323_0341_D.MP4`
(NORMAL 300/300, light aerial traffic, the project's one native drone
source). This is not a measured false-positive *rate* against ground
truth -- there is no frame-level congestion annotation for any of these
clips, only content/title-level judgment of "known non-congested." `vid3`
is a locked-test source (`datasets/vietnam_dataset_v5/manifest.csv`:
`source_id=vid3, split=test`); confidence=0.1 was already fixed from run42
(a train-split clip) before this check, so this is reported as a one-time
confirmatory result on that source, not used to further tune the
threshold, consistent with [the dataset protocol](dataset_protocol.md)'s
rule against selecting on the locked test.

**Provenance note.** The pipeline config backing run42/run43 was edited in
place between runs (`fixed_camera` -> `uav_motion`) to produce run43, so
its current content no longer reproduces run42; `run44`'s config was
likewise reused (not a fresh file) before `run45` used a separate file.
Every run's own `run.json` still records its exact parameters, so no
individual result is in doubt, but the shared config files are not a
frozen, rerunnable record on their own. A single hash-backed manifest
covering all 7 runs (model, source, and per-run parameters/results) is
committed at `experiments/lowconf_congestion_ab_20260820/run.json` as the
authoritative record; `output/pipeline/run42`-`run48` themselves are not
committed (`output/` is gitignored).

**Recommendation:** lowering detector confidence (fixed_camera mode,
occupancy corroboration intact) is a safe, real, but partial improvement --
worth adopting where a scene's known failure mode is under-triggering, not
worth treating as solved. The `uav_motion` count-bypass is rejected with
concrete evidence, not merely untried. The 73%-of-clip gap that confidence
alone does not close still points at the same two candidates above (a
dedicated low-confidence density pass, or a better texture feature) as the
remaining real next step, now with a clearer picture of what does and does
not move the needle.
