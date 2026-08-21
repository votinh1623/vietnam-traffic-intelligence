# Vietnam Traffic Intelligence

Leakage-controlled traffic detection, tracking, counting, alerting, and
multimodal reporting for UAV traffic video, developed and measured on an
NVIDIA RTX 3050 Laptop GPU (6 GB VRAM). Quantization and physical edge/NPU
deployment are explicitly deferred until the current research goal is
complete.

See [docs/quickstart.md](docs/quickstart.md) for installation and CLI usage.

---

## Why this project

Vietnamese road scenes are dense, dominated by small motorcycles and
pedestrians, and frequently affected by occlusion and camera motion. A useful
system must do more than draw boxes: it must preserve identities, aggregate
traffic state, explain noteworthy events, and report the accuracy and runtime
cost of every optimization honestly.

## Research objectives

The current goal is complete only when the project demonstrates and evaluates
all of the following, without relying on quantization or physical deployment:

1. detect and count vehicles from UAV video;
2. improve counting reliability with tracking and explicit handling of
   detection/tracking errors;
3. use pretrained VLM/LLM models to generate a traffic description;
4. emit alerts for high density or a narrowly defined abnormal event; and
5. evaluate the system on real UAV data, primarily VisDrone.

| Goal | Maps to objective(s) | Required evidence for completion |
|---|---|---|
| Detection | 1 | Standard-versus-sliced inference on VisDrone-DET with overall, per-class, object-scale, and latency metrics |
| Tracking | 2 | Class-aware IDF1, MOTA, MOTP distance, ID switches, and fragmentations on VisDrone-MOT |
| Counting | 1, 2 | Ground-truth trajectory-derived line-crossing counts and error metrics; comparison against the selected tracker output |
| Alerts | 4 | Deterministic high-density alert plus explicitly configured wrong-way or prolonged-stop event, with synthetic tests and video evidence |
| VLM/LLM description | 3 | At least one end-to-end report from pretrained models with structured-event numbers kept separate from visual claims |
| UAV system evaluation | 5 | Reproducible detector, tracker, counting, alert, and end-to-end results with model/config/data hashes |

SAHI is evaluated first as an inference-only small-object method on VisDrone;
it is not assumed to improve tracking until measured. The consumed Vietnam v5
locked test remains final and is not used to select slicing parameters.

## System architecture

```text
camera / UAV / video
        |
        v
decode and frame sampling
        |
        v
YOLOv8 detector ---> ByteTrack ---> traffic analytics ---> structured events
                                              |                    |
                                              v                    v
                                      selected frame/clip ---> VLM
                                                                   |
                                             structured events ----+
                                                                   v
                                                            LLM reasoning
                                                                   |
                                                                   v
                                                   dashboard / alert / API
```

The LLM and VLM sit outside the per-frame critical path, invoked only on
selected events or at a configured interval. Component boundaries are
detailed in [the multimodel architecture](docs/multimodel_architecture.md).

## Dataset

The original `vietnam_dataset_v2` split leaked all 12 source videos across
train/validation/test; its metrics are historical only. **Vietnam v5**
supersedes it: source-disjoint splits, polygon-to-box conversion, 53 exact
duplicate boxes removed, content-addressed test lock. Full audit, source
provenance, and the object-scale root-cause analysis behind the detection
gap below are in [the dataset protocol](docs/dataset_protocol.md).

| Split | Images | Bus | Car | Motorcycle | Pedestrian | Truck | Total boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 819 | 1,559 | 12,439 | 29,005 | 3,265 | 2,781 | 49,049 |
| Calibration | 111 | 158 | 742 | 4,360 | 1,259 | 23 | 6,542 |
| Validation | 108 | 75 | 872 | 2,554 | 2,462 | 60 | 6,023 |
| Locked test | 176 | 622 | 6,038 | 4,486 | 170 | 327 | 11,643 |
| **Total** | **1,214** | **2,414** | **20,091** | **40,405** | **7,156** | **3,191** | **73,257** |

Key limitation: only 12 total source videos (11 repurposed YouTube uploads,
1 native drone capture); calibration has 23 truck boxes, locked test has
170 pedestrian boxes. Raw videos and generated datasets are excluded from
Git.

## Evaluation policy

- Select checkpoints and thresholds on validation; calibration is PTQ-only.
- Never use the locked test for model, tracker, prompt, or backend selection.
- Report precision, recall, mAP50, and mAP50-95 per class and overall.
- Report tracking only after sequence-level evaluator validation.
- Split latency into preprocessing, inference, postprocessing, tracking, VLM,
  and LLM stages; report hardware, precision, warm-up, sample count, and
  percentile latency.
- Treat smoke runs and legacy leaked runs as diagnostics, never final
  evidence.

Deployment evaluation is not yet executed. Its protocol -- FP32/FP16/INT8/
TensorRT candidate requirements, VLM/LLM candidate requirements, calibration
split discipline -- is fixed in advance in
[the benchmark protocol](docs/benchmark_protocol.md).

## Results

Each subsection: setup, measured metrics, the hashed run record, and what the
result does/does not prove. Full method detail and rationale for every item
below live in [the benchmark protocol](docs/benchmark_protocol.md); this
section states outcomes only.

### Detection

YOLOv8s: COCO -> VisDrone2019-DET (`mAP50=0.389` @ epoch 74) -> Vietnam v5
(30 epochs, 1280, full weights, checkpoint selected on validation
`mAP50-95`). Why YOLOv8 over newer Ultralytics generations:
[benchmark protocol](docs/benchmark_protocol.md#why-yolov8).

| Split | Precision | Recall | mAP50 | mAP50-95 | Use |
|---|---:|---:|---:|---:|---|
| Validation | 0.762 | 0.504 | 0.600 | 0.344 | Checkpoint selection |
| Locked test | 0.215 | 0.287 | 0.148 | 0.062 | Final reporting only |

Small-object inference-mode selection (548 VisDrone-DET val images, COCO-style AP):

| Mode | AP | AP-small | p50 latency (ms/img) | Decision |
|---|---:|---:|---:|---|
| Standard 640 | 0.212 | 0.118 | 23.7 | Reference |
| **Standard 1280** | **0.264** | **0.194** | 130.8 | **Selected mode** |
| SAHI 640 tiles | 0.193 | 0.142 | 282.0 | Rejected |
| Hybrid full-frame + tiles | 0.177 | 0.117 | 200.1 | Rejected |

**Checkpoint promoted 2026-08-21** (superseding the 640-trained checkpoint
above for UAV pipeline configs): a 5-epoch native-1280 continuation, gated
at AP-small +0.010 absolute / overall AP drop <=0.005, passed with margin --
AP 0.264->0.296 (+0.0325), AP-small 0.194->0.216 (+0.0223) -- and propagated
into a real tracking gain (see [Tracking](#tracking)). Full writeup:
[benchmark protocol](docs/benchmark_protocol.md#visdrone-highres-fine-tune-pilot-and-checkpoint-promotion).

Two further ablations targeted the validation-to-test gap directly and were
**rejected** (full method, incident history, and root-cause discussion:
[benchmark protocol](docs/benchmark_protocol.md#detector-training-and-validation)):

| Split | Precision | Recall | mAP50 | mAP50-95 | Result |
|---|---:|---:|---:|---:|---|
| Locked test, baseline (CIoU) | 0.215 | 0.287 | 0.148 | 0.062 | Selected |
| Locked test, NWD loss (alpha=0.5, C=16) | 0.194 | 0.251 | 0.128 | 0.054 | **Rejected** |
| Locked test, P2 head (stride-4) | 0.146 | 0.206 | 0.095 | 0.037 | **Rejected** |

**Evidence.** `experiments/yolov8s_v5_locked_test_20260818/run.json`,
`experiments/visdrone_det_small_object_v1_20260818/run.json`,
`experiments/yolov8s_v5_seed0_nwd_20260819T094236/run.json`,
`experiments/yolov8s_v5_seed0_p2_continued_20260819T152917/run.json`,
`experiments/visdrone_highres_pilot_and_reid_results_20260821/run.json`.

**Caveat.** The validation-to-test gap is a real object-scale/source shift,
not an evaluation bug; v5 is a functional prototype, not a
production-general detector. NWD and P2 both targeted this gap and failed,
consistent with it being primarily a training-data scarcity problem rather
than something a loss or architecture change alone corrects -- see the
[dataset protocol](docs/dataset_protocol.md#source-composition-and-the-object-scale-gap)
for the root-cause analysis.

### Tracking

Two controlled comparisons, different detector/class scopes -- compare
modes within each table, not across tables. The algorithm/ReID table uses
the Vietnam-v5 five-class checkpoint (1280/confidence 0.4); the resolution
table uses the VisDrone ten-class checkpoint filtered to eight vehicle
classes (confidence 0.1). Both on all 2,846 frames across 7
VisDrone2019-MOT-val sequences, class-aware IoU matching at 0.5.

| Metric | ByteTrack (baseline) | BoT-SORT | BoT-SORT+ReID |
|---|---:|---:|---:|
| IDF1 | 0.309 | 0.355 | 0.358 |
| MOTA | 0.020 | 0.005 | 0.004 |
| ID switches | 462 | **207** | 209 |
| Fragmentations | 1,491 | 1,673 | 1,673 |
| HOTA | 0.288 | 0.322 | 0.324 |
| DetA | 0.196 | 0.206 | 0.206 |
| AssA | 0.453 | 0.535 | 0.541 |

Resolution comparison (640 vs. 1280, vehicle classes only, ByteTrack):

| Mode | IDF1 | MOTA | Precision | Recall | ID switches |
|---|---:|---:|---:|---:|---:|
| Standard 640 | 0.473 | **0.215** | **0.663** | 0.449 | **411** |
| Standard 1280 | **0.481** | 0.132 | 0.578 | **0.521** | 568 |

**Evidence.** `experiments/tracking_visdrone_mot_val_v1_20260818/run.json`,
`experiments/tracking_visdrone_mot_resolution_v1_20260818/run.json`,
`experiments/tracking_hota_corrected_20260820/run.json` (committed,
hash-backed corrected-HOTA record; `benchmark_outputs/` paths are local and
gitignored).

**Caveat.** Provenance-controlled integration baseline, not an official
VisDrone benchmark or Vietnam-domain evidence. `scripts/evaluate_hota.py`
had a bug silently dropping zero-prediction frames instead of scoring them
as false negatives (32-33 frames out of 2,846 per tracker); fixed, moved
every number by <=0.002, does not change any conclusion here. DetA is
nearly flat across trackers (0.196-0.206) while AssA moves with tracker
choice -- detection recall, not association, is this pipeline's dominant
limitation. Full BoT-SORT/ReID ablation detail, the ReID-embedding
reconfirmation, and a rejected aligned-threshold ByteTrack candidate:
[benchmark protocol](docs/benchmark_protocol.md#bot-sort-and-reid-ablation).

### Counting

Production analytics state machine against native VisDrone MOT ground-truth
trajectories, 2,382 frames across 6 traffic sequences, three frozen
horizontal counting lines.

| Tracking profile | Frame-count macro MAE (veh/frame) | Frame-count micro WAPE | Crossing WAPE |
|---|---:|---:|---:|
| Standard 640 | 10.45 | 0.372 | 0.593 |
| **Standard 1280** | **9.80** | **0.319** | **0.560** |

**Evidence.** `experiments/counting_visdrone_mot_v1_20260818/run.json`.

**Caveat.** Crossing WAPE of 0.560 demonstrates measurable counting, not
production-grade accuracy -- no stabilization or BEV transform is applied,
so image-space crossings measure agreement with GT trajectories, not
calibrated physical flow.

### Alerts

Deterministic `prolonged_stop` alert (synthetic-tested) plus the congestion
state machine, accepted on two real demo clips.

| Clip | Bbox-union occupancy median (range) | State timeline |
|---|---:|---|
| `traffic_jam.mp4` | 0.589 (0.488-0.724) | `NORMAL` to `CONGESTED` at 2.12 s; 230 congested frames |
| `traffic_normal.mp4` | 0.135 (0.069-0.219) | `NORMAL` for 300/300 frames |

**Evidence.** `experiments/alerts_acceptance_v1_20260818/run.json`.

**Caveat.** Two-clip qualitative acceptance, not precision/recall evidence;
thresholds were tuned on these same two scenes. No labeled real
abnormal-stop clip exists, so real prolonged-stop accuracy is unmeasured.

### VLM/LLM description

Two-stage pretrained contract, no fine-tuning: Qwen3-VL-2B-Instruct (visual
assessment) feeds Qwen3-0.6B (Vietnamese report). Numeric facts are
assembled from the deterministic event, not generated, and validated
automatically against it.

| Check | Result |
|---|---|
| Contract validity | Valid on all completed runs; altered/invented numeric facts rejected automatically |
| Grounded description (v3 prompt) | Correctly named the dominant vehicle type on two distinct real clips |
| Formal quality / hallucination rate | Not yet measured (no frozen human-annotated result) |

**Evidence.** `output/reasoning/adhoc/run32-vlm-v3prompt.json`,
`output/reasoning/adhoc/run34-vlm-v3prompt.json` (ad hoc, not frozen
experiments).

**Caveat.** Every run before v3 copied a literal example sentence from the
prompt instead of describing the actual image -- fixed; full writeup:
[reasoning protocol](docs/reasoning_protocol.md#prompt-copying-bug-v1-to-v3).
`validate_grounding_policy` still does not show clip frames to the VLM
despite the name -- an open gap. Human reference annotations for formal
quality scoring are pending.

### UAV system evaluation

Full pipeline (selected detector + ByteTrack + analytics + evidence export)
run end to end on 300 real 1080p UAV frames. Re-run 2026-08-21 after
promoting the highres-pilot checkpoint; original 2026-08-18 numbers kept for
comparison.

| Metric | 2026-08-18 (640-trained) | 2026-08-21 (highres pilot) |
|---|---:|---:|
| End-to-end throughput | 3.70 FPS | 3.40-3.46 FPS (normal run-to-run variance, not a new bottleneck) |
| Congestion, fixed camera ROI | **Failed** -- stayed `NORMAL` for all 300 frames | **Still fails the same way** -- see caveat |
| Congestion, `uav_motion` mode | **Fixed** -- `NORMAL`->`CONGESTED` at frame ~64-65 | **Reconfirmed** -- same transition frame |

**Evidence.** `experiments/uav_pipeline_e2e_v1_20260818/run.json`,
`experiments/visdrone_highres_pilot_and_reid_results_20260821/run.json`
(promotion record); `output/pipeline/run52`/`run53` (re-run, gitignored).
Full GMC A/B and re-confirmation detail:
[benchmark protocol](docs/benchmark_protocol.md#uav-moving-camera-analytics-gmc).

**Caveat.** The `NORMAL`->`CONGESTED` fix is `analytics.mode: uav_motion`'s
count-alone trigger, not GMC -- an A/B test shows the same transition frame
with or without GMC. The `fixed_camera` failure is a mode/threshold
limitation on this clip, not a detection-recall gap, so it did not change
when the detector improved. GMC's own positional-accuracy value (does the
re-projected ROI track a fixed real-world region under pan/zoom) is
unverified. This remains one real clip, not a benchmark across multiple UAV
sources.

## Known limitations

- Source videos collected from the web have incomplete provenance and cannot
  be assumed redistributable.
- Dataset remains small and class-imbalanced; only 12 source videos total,
  so which sources land in test measurably changes results. Cross-source
  validation (leave-one-source-out) was deliberately not run (many GPU-hours
  for a variance whose cause is already known and disclosed). Treat every
  locked-test number as one sample from an unmeasured distribution.
- The tracking result is an integration baseline on VisDrone, not a
  Vietnam-domain or official VisDrone benchmark.
- Traffic speed requires camera calibration or a documented approximation.
  Bbox-union occupancy is image-plane box coverage, not physical road
  occupancy.
- Line-crossing counts depend on stable ByteTrack identities; the
  occlusion-driven error rate has not been measured on the two demo videos.
- VLM/LLM quality, hallucination rate, and quantization effects are not yet
  measured.
- Global motion compensation is 2D image-plane alignment only, not BEV or
  GPS/IMU-based, and can lose lock under hard cuts/fast pans/low-texture
  frames -- check `gmc_total_failures` (run-wide), not just
  `gmc_consecutive_failures_at_end` (end-of-run streak only). See
  [UAV system evaluation](#uav-system-evaluation) for the GMC-vs-`uav_motion`
  distinction.
- Congestion detection depends on the detector resolving individual boxes.
  Under severe occlusion (a tightly packed, stalled crowd), detector recall
  collapses exactly when congestion is worst -- observed directly on a real
  rush-hour clip where the detector drew zero boxes over a gridlocked mass.
  A detection-independent stillness signal was built and tested as an
  automatic trigger; **rejected and root-caused** (Laplacian texture cannot
  distinguish packed vehicles from any other static detailed surface). A
  **visual heatmap** variant of the same signal (decoupled from the state
  machine) **works** and is real-pipeline-validated. Lowering detector
  confidence is a safe, partial fix; dropping the occupancy co-requirement
  to let count alone trigger `CONGESTED` was tried and **rejected** (79%
  false-positive rate on a light-traffic reference clip). Full trail:
  [benchmark protocol](docs/benchmark_protocol.md#detection-independent-stillness-signal-prototype).
- The dashboard's live-frame write can fail on Windows due to transient file
  locks (handled as non-fatal; the dashboard can show a stale frame).
- No model has yet been benchmarked on a physical edge NPU.

## Roadmap

Current delivery priority is the deterministic CV product. Reasoning work is
limited to pretrained-model integration and demo quality; VLM/LLM
fine-tuning, all quantization work, and physical deployment are explicitly
deferred. Full method detail for every completed item is in
[the benchmark protocol](docs/benchmark_protocol.md) unless linked otherwise.

**Dataset and detector**
- [x] Audit the legacy dataset, identify leakage, and build source-grouped, hash-locked v5 splits.
- [x] Complete v5 fine-tuning, validation-based checkpoint selection, and the one-time locked-test evaluation.
- [x] Compare standard, high-resolution, SAHI, and hybrid inference on VisDrone-DET; select standard 1280.
- [x] Test an NWD bbox-loss ablation against the small-object gap -- **rejected**.
- [x] Test a P2 detection-head architecture ablation against the same gap -- **rejected**, including a training-crash-and-recovery incident.
- [x] Diagnose a train/infer resolution mismatch on the VisDrone baseline checkpoint (trained at 640, infers at 1280) and VRAM-validate a gated continuation pilot.
- [x] Run the 5-epoch native-1280 continuation, evaluate against the frozen gate -- **passed with margin**, promoted as the UAV pipeline default 2026-08-21.
- [ ] Deferred: evaluate a newer Ultralytics architecture generation (e.g. YOLO26) as a new baseline.
- [ ] Deferred: export and benchmark detector FP16/INT8 candidates.

**Tracking and counting**
- [x] Feed the selected detector into ByteTrack once per source frame; compare 640 vs. 1280.
- [x] Repair and validate sequence-level class-aware tracking evaluation (motmetrics IoU-distance fix).
- [x] Integrate TrackEval for HOTA/DetA/AssA; decompose the bottleneck as detection-limited, not association-limited.
- [x] Test a BoT-SORT/ReID ablation -- algorithm switch helped, `model:auto` ReID did not.
- [x] Test a real pretrained ReID embedding (`yolo26n-reid.onnx`) against `model:auto` -- **no improvement**, reconfirms the detection-recall bottleneck.
- [x] Derive frame-count and line-crossing ground truth from VisDrone-MOT trajectories and measure error.

**Alerts and analytics**
- [x] Implement deterministic analytics/event schema with synthetic tests; complete ROI/counting-line/congestion acceptance on two demo videos.
- [x] Add and synthetic-test a prolonged-stop alert with speed hysteresis and gap reset.
- [x] Diagnose the UAV camera-motion ROI failure and implement GMC (`analytics.mode: uav_motion`, `gmc_enabled`) -- the fix is `uav_motion`'s count-alone trigger, not GMC itself (A/B tested); reconfirmed on the promoted checkpoint.
- [x] Build a detection-independent stillness signal for severe-occlusion jams -- automatic trigger **rejected and root-caused**; visual heatmap variant **works**.
- [x] Test two cheap congestion-trigger fixes (lower confidence: partial, safe fix; count-alone trigger: **rejected**, false-positives on a light-traffic clip).
- [ ] Open product gap: the confidence fix alone still leaves the motivating clip mostly `NORMAL`. Also open: per-lane/multi-region ROI decomposition and verified ego-motion compensation.

**VLM/LLM and evidence**
- [x] Freeze VLM/LLM evaluation inputs, JSON/prompt contract v1, two-reviewer annotation tooling; complete two independent reviewer sets for reasoning eval v1.
- [ ] Resolve or formally defer the reasoning adjudication queue; does not block CV delivery.
- [x] Fix the VLM/LLM prompt-copying bug (v1 to v3), verified grounded on two real clips.
- [x] Add deterministic event keyframe/clip evidence selection with provenance hashes.
- [x] Add a Streamlit dashboard over pipeline run output (headless boot verified).

**TVLR (paused after Stage B)**
- [x] Freeze the offline TVLR feasibility protocol (excludes detections ByteTrack already recovers, protects an internal holdout, forbids a real-time claim). See [TVLR protocol](docs/tvlr_protocol.md).
- [x] Run the Stage-B development oracle on 896 VisDrone-MOT frames: 33.8% of ByteTrack-missed tiny/occluded GT recoverable, +0.146 recall ceiling, only 6.1% WAPE improvement and worse on one of three dev sequences -- real opportunity, but false-positive control is the central Stage-C risk. Not an achieved result. `experiments/tvlr_oracle_dev_v1_20260820/run.json`.
- [ ] **Paused**: Stage C (implementation against frozen baselines) is deprioritized while detector/tracking work on VisDrone continues instead.

**Deferred beyond current goal**
- [ ] Quantize and benchmark the selected VLM and LLM.
- [ ] Validate an appropriate physical edge/NPU target.

## License

Code in this repository is licensed under
[GNU AGPL-3.0-only](LICENSE). Dataset, video, pretrained-weight, and third-party
asset licenses remain separate and must be verified before redistribution or
commercial use.

## Acknowledgements

This project uses Ultralytics YOLO, PyTorch, OpenCV, pandas, motmetrics,
VisDrone, Roboflow-assisted labeling, and ByteTrack concepts. Physical NPU
results from third-party projects are not treated as evidence for this system;
all deployment claims in this repository require measurements from its own
artifacts and declared hardware.
