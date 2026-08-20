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
it is not assumed to improve tracking until merged full-frame detections have
been passed once per source frame to ByteTrack and measured. The consumed
Vietnam v5 locked test remains final and is not used to select slicing
parameters.

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

The LLM and VLM are intentionally outside the per-frame critical path. They
are invoked on selected events or at a configured interval to limit latency,
memory use, and hallucination surface. Component boundaries are detailed in
[the multimodel architecture](docs/multimodel_architecture.md).

## Dataset

The original `vietnam_dataset_v2` split contained severe temporal and source
leakage: all 12 source videos appeared across train, validation, and test.
Its metrics are preserved as historical evidence only and are invalid for
scientific comparison. Vietnam v5 supersedes it, materialized
non-destructively with source-disjoint splits, polygon-to-box conversion,
deterministic removal of 53 exact duplicate boxes, and a content-addressed
test lock.

| Split | Images | Bus | Car | Motorcycle | Pedestrian | Truck | Total boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 819 | 1,559 | 12,439 | 29,005 | 3,265 | 2,781 | 49,049 |
| Calibration | 111 | 158 | 742 | 4,360 | 1,259 | 23 | 6,542 |
| Validation | 108 | 75 | 872 | 2,554 | 2,462 | 60 | 6,023 |
| Locked test | 176 | 622 | 6,038 | 4,486 | 170 | 327 | 11,643 |
| **Total** | **1,214** | **2,414** | **20,091** | **40,405** | **7,156** | **3,191** | **73,257** |

Important limitations: calibration contains only 23 truck boxes; validation
contains 75 bus and 60 truck boxes; locked test contains only 170 pedestrian
boxes; four source-unknown frames and 74 conflicting duplicate-frame
annotation groups were excluded; a visual near-overlap audit (dHash distance
12/256) found no candidate overlap between renamed-YouTube sources, but this
does not prove ownership or redistribution rights. Raw videos and generated
datasets are intentionally excluded from Git; full audit detail is in
[the dataset protocol](docs/dataset_protocol.md).

**Source composition is the likely root cause of the object-scale gap
documented under [Detection](#detection).** All 1,214 images come from only
11 source videos (5 train / 2 calibration / 2 validation / 3 test): 10 are
repurposed YouTube uploads and exactly 1 (`DJI_20250516071323_0341_D`, in
validation) is a native drone capture. The two sources explicitly titled as
aerial drone footage both landed in the locked test by the source-disjoint
split -- with only 11 sources total, this is close to unavoidable, not a
labeling defect. Measured directly from the label files (sqrt(w*h) in
pixels at imgsz=1280): only 4.7% of train boxes are under 16px versus 48.8%
of test boxes. A model trained on this data sees real small-object examples
rarely, then is evaluated on a split where they dominate. This is a data
scarcity problem, not something a loss function or architecture change can
fully correct (see the NWD ablation and P2 head ablation under
[Detection](#detection), both of which train on this same imbalance).

## Evaluation policy

- Select checkpoints and thresholds on validation; use calibration only for
  PTQ calibration and related configuration.
- Never use the locked test for model, tracker, prompt, or backend selection.
- Report precision, recall, mAP50, and mAP50-95 per class and overall.
- Report tracking only after sequence-level evaluator validation.
- Split latency into preprocessing, inference, postprocessing, tracking, VLM,
  and LLM stages; report hardware, precision, warm-up, sample count, and
  percentile latency.
- Treat smoke runs and legacy leaked runs as diagnostics, never final
  evidence.

Deployment evaluation is not yet executed (no quantized/edge candidate has
been produced), but its protocol is fixed in advance: detector FP32/FP16/
INT8/TensorRT candidates need accuracy, per-stage latency, throughput, VRAM,
and size; VLM/LLM candidates need task quality, latency, memory, and size;
INT8 calibration must use the calibration split only. The full measurement
contract is defined in [the benchmark protocol](docs/benchmark_protocol.md).

## Results

Each subsection follows the required-evidence row above: setup, measured
metrics, the hashed run record, and the boundary of what the result does and
does not prove.

### Detection

**Why YOLOv8.** Chosen over newer Ultralytics releases (e.g. YOLO26, which
ships in the same pinned `ultralytics==8.4.115` install used here) for three
reasons, not because it was benchmarked as more accurate: it is a widely
validated baseline with a large body of independent small-object/aerial
literature to contextualize results against (including the NWD paper this
project's own loss ablation is based on); it has years of mature ONNX/
TensorRT/edge export tooling, relevant to this project's deferred edge/NPU
goal; and its internals are stable and well documented enough to safely
monkey-patch (this project's NWD loss) and extend (the P2 head ablation)
directly, which is a real risk to redo correctly on a newer, less-verified
architecture. Evaluating a newer architecture generation as a new baseline
is a legitimate, identified future direction -- not yet executed, and not
assumed to be worse or better than YOLOv8 until it is.

**Setup.** YOLOv8s initialized from COCO, fine-tuned on VisDrone2019-DET
(`mAP50=0.389` at epoch 74) as a checkpoint, then fine-tuned again on the
source-disjoint Vietnam v5 dataset: full weights (`freeze=0`), 30 epochs,
input 1280, batch 4, AdamW at learning rate 0.0005, seed 0, checkpoint
selected on validation `mAP50-95` only (epoch 29).

| Split | Precision | Recall | mAP50 | mAP50-95 | Use |
|---|---:|---:|---:|---:|---|
| Validation | 0.762 | 0.504 | 0.600 | 0.344 | Checkpoint selection |
| Locked test | 0.215 | 0.287 | 0.148 | 0.062 | Final reporting only |

Small-object inference selection (separate frozen comparison, all 548
VisDrone2019-DET validation images, four inference modes, COCO-style AP):

| Mode | AP | AP-small | p50 latency (ms/img) | Decision |
|---|---:|---:|---:|---|
| Standard 640 | 0.212 | 0.118 | 23.7 | Reference |
| **Standard 1280** | **0.264** | **0.194** | 130.8 | **Selected for tracking pipeline** |
| SAHI 640 tiles | 0.193 | 0.142 | 282.0 | Small-object ablation only |
| Hybrid full-frame + tiles | 0.177 | 0.117 | 200.1 | Rejected |

NWD bbox-loss ablation (ties the loss directly to the object-scale gap
above; identical dataset/init/hyperparameters as the baseline, only the
bbox loss differs -- see [benchmark protocol](docs/benchmark_protocol.md#nwd-bbox-loss-ablation-rejected)
for how the two runs were verified hash-identical apart from that):

| Split | Precision | Recall | mAP50 | mAP50-95 | Result |
|---|---:|---:|---:|---:|---|
| Locked test, baseline (CIoU) | 0.215 | 0.287 | 0.148 | 0.062 | Selected |
| Locked test, NWD (alpha=0.5, C=16) | 0.194 | 0.251 | 0.128 | 0.054 | **Rejected -- worse on every metric and every class** |

P2 detection-head ablation (adds a 4th Detect scale at stride 4, so the
network has an output receptive field sized for the smallest objects; see
[benchmark protocol](docs/benchmark_protocol.md#p2-detection-head-ablation-rejected)
for the training-crash-and-recovery history and why this checkpoint is a
two-stage/restarted fine-tune, not a single clean run like baseline/NWD):

| Split | Precision | Recall | mAP50 | mAP50-95 | Result |
|---|---:|---:|---:|---:|---|
| Locked test, baseline (CIoU) | 0.215 | 0.287 | 0.148 | 0.062 | Selected |
| Locked test, P2 (stride-4 head) | 0.146 | 0.206 | 0.095 | 0.037 | **Rejected -- worse than baseline and worse than NWD on every metric** |

**Evidence.** `experiments/yolov8s_v5_locked_test_20260818/run.json`,
`experiments/visdrone_det_small_object_v1_20260818/run.json`,
`experiments/yolov8s_v5_seed0_nwd_20260819T094236/run.json`,
`benchmark_outputs/detector_v5_nwd_locked_test/run.json`,
`experiments/yolov8s_v5_seed0_p2_continued_20260819T152917/run.json`,
`benchmark_outputs/detector_v5_p2_locked_test/run.json`.

**Caveat.** The validation-to-test gap is real and driven by an
object-scale/source shift (82.9%-100% of locked-test boxes cover under 0.1%
of the image, by class, versus 1.3%-70.0% on validation), not an evaluation
bug. V5 is a functional prototype, not a production-general detector. The
historical `vietnam_dataset_v2` result (`mAP50=0.745`) is excluded here
because it is invalid for scientific comparison (split leakage). The NWD
attempt at closing that gap failed at these settings, plausibly because its
constant was tuned to the *test* box-size distribution while the loss trains
on *train* boxes roughly 3x larger, saturating the similarity term near zero
for most training data -- untested variants (train-scaled constant, lower
alpha, epoch-scheduled alpha) might still work; none have been run. The P2
architecture change performed even worse: its validation-to-test drop
(0.321->0.037, ~8.8x) is proportionally larger than both baseline (~5.5x)
and NWD (~6.2x), despite directly targeting this exact gap. Both the P2
architecture's design rationale and NWD's constant were chosen with
knowledge of the locked test's box-size distribution, so neither result is a
blind confirmatory test -- both are exploratory. Both a loss-side fix (NWD)
and an architecture-side fix (P2) have now failed on this same gap, which is
consistent with the gap being primarily a training-data scarcity problem
(see the Dataset section) rather than something either change alone can
correct. Full per-class diagnosis and the P2 training incident history:
[benchmark protocol](docs/benchmark_protocol.md).

### Tracking

**Setup.** ByteTrack over the selected VisDrone checkpoint, all 2,846 frames
across 7 VisDrone2019-MOT-val sequences, class-aware IoU matching at 0.5.

| Metric | ByteTrack (baseline) | BoT-SORT | BoT-SORT+ReID |
|---|---:|---:|---:|
| IDF1 | 0.309 | 0.355 | 0.358 |
| MOTA | 0.020 | 0.005 | 0.004 |
| ID switches | 462 | **207** | 209 |
| Fragmentations | 1,491 | 1,673 | 1,673 |
| HOTA | 0.288 | 0.323 | 0.325 |
| DetA | 0.197 | 0.207 | 0.207 |
| AssA | 0.453 | 0.536 | 0.541 |

Resolution comparison (640 vs. 1280, vehicle classes only, same sequences,
ByteTrack):

| Mode | IDF1 | MOTA | Precision | Recall | ID switches |
|---|---:|---:|---:|---:|---:|
| Standard 640 | 0.473 | **0.215** | **0.663** | 0.449 | **411** |
| Standard 1280 | **0.481** | 0.132 | 0.578 | **0.521** | 568 |

**Evidence.** `experiments/tracking_visdrone_mot_val_v1_20260818/run.json`,
`experiments/tracking_visdrone_mot_resolution_v1_20260818/run.json`,
`benchmark_outputs/tracking_visdrone_mot_botsort/{run.json,hota.json}`,
`benchmark_outputs/tracking_visdrone_mot_botsort_reid/{run.json,hota.json}`.

**Caveat.** This is a provenance-controlled integration baseline, not an
official VisDrone benchmark (no ignore-region handling) or Vietnam-domain
evidence. HOTA's DetA/AssA decomposition (now integrated via TrackEval,
see [benchmark protocol](docs/benchmark_protocol.md)) shows DetA is nearly
identical across all three trackers (0.197-0.207) while AssA moves with the
tracker choice -- i.e. detection recall, not association, is this pipeline's
dominant limitation, which the BoT-SORT/ReID ablation cannot fix by itself.
Switching ByteTrack to BoT-SORT nearly halved ID switches and raised IDF1
and AssA, but ReID (`model:auto`, reusing the detector's own pre-Detect-head
features, no separate ReID model) added essentially nothing on top of that
(IDF1 0.355->0.358, ID switches 207->209 -- within noise), and MOTA/
fragmentations got slightly worse under BoT-SORT. `bytetrack_custom.yaml`
remains the pipeline default pending a decision on this trade-off. A
ByteTrack candidate with aligned 0.4 track/new thresholds was also tested
and rejected (worse IDF1, MOTA, and +257 ID switches). Full numbers:
[benchmark protocol](docs/benchmark_protocol.md).

### Counting

**Setup.** The production analytics state machine run against native
VisDrone MOT ground-truth trajectories, 2,382 frames across 6 traffic
sequences (basketball-court sequence excluded), at three frozen horizontal
counting lines.

| Tracking profile | Frame-count macro MAE (veh/frame) | Frame-count micro WAPE | Crossing WAPE |
|---|---:|---:|---:|
| Standard 640 | 10.45 | 0.372 | 0.593 |
| **Standard 1280** | **9.80** | **0.319** | **0.560** |

**Evidence.** `experiments/counting_visdrone_mot_v1_20260818/run.json`.

**Caveat.** Standard 1280 is selected as the quality-first counting profile,
but crossing WAPE of 0.560 demonstrates measurable counting, not
production-grade accuracy — these UAV cameras move/zoom and no
stabilization or BEV transform is applied, so image-space crossings measure
agreement with GT trajectories, not calibrated physical flow. Full numbers:
[benchmark protocol](docs/benchmark_protocol.md).

### Alerts

**Setup.** Deterministic `prolonged_stop` alert (entry continuity,
release-speed hysteresis, tracking-gap reset — all synthetic-tested) plus
the congestion state machine, accepted on two real demo clips.

| Clip | Bbox-union occupancy median (range) | State timeline |
|---|---:|---|
| `traffic_jam.mp4` | 0.589 (0.488-0.724) | `NORMAL` to `CONGESTED` at 2.12 s; 230 congested frames |
| `traffic_normal.mp4` | 0.135 (0.069-0.219) | `NORMAL` for 300/300 frames |

**Evidence.** `experiments/alerts_acceptance_v1_20260818/run.json`.

**Caveat.** This is a two-clip qualitative acceptance result, not
precision/recall evidence, and thresholds were tuned against these same two
scenes — they are not shown to generalize to another camera or viewpoint.
No labeled real abnormal-stop clip exists, so real prolonged-stop accuracy
remains unmeasured.

### VLM/LLM description

**Setup.** Two-stage pretrained contract, no fine-tuning: Qwen3-VL-2B-Instruct
(visual assessment) feeds Qwen3-0.6B (Vietnamese report). Numeric facts in
the report are assembled from the deterministic event, not generated —
validated automatically against the source event.

| Check | Result |
|---|---|
| Contract validity | Valid on all completed runs; altered/invented numeric facts are rejected automatically |
| Grounded, non-generic description (v3 prompt, after fixing a copy-paste bug — see caveat) | Correctly named the dominant vehicle type on two distinct real clips, matching real analytics counts |
| Formal quality / hallucination rate | Not yet measured (no frozen human-annotated evaluation result yet) |

**Evidence.** `output/reasoning/adhoc/run32-vlm-v3prompt.json`,
`output/reasoning/adhoc/run34-vlm-v3prompt.json` (ad hoc verification runs,
not a frozen experiment record). Reasoning evaluation v1 input lock:
`ecfd9a1e44ae1be4991f5e87dbf65d3ce9e42c4185a00db782e728258673d18b`.

**Caveat.** Every run before this fix copied a literal example sentence out
of the prompt instead of describing the actual image, regardless of image
content — including once producing a factually wrong answer on a
truck-dominated scene. Full writeup:
[reasoning protocol](docs/reasoning_protocol.md#prompt-copying-bug-v1-to-v3).
`validate_grounding_policy` also still does not show clip frames to the VLM
despite the policy name — a real, open gap. Human reference annotations for
formal quality scoring are still pending.

### UAV system evaluation

**Setup.** Full pipeline (selected 1280 detector + ByteTrack + analytics +
evidence export) run end to end on 300 real 1080p UAV frames.

| Metric | Value |
|---|---:|
| End-to-end throughput | 3.70 FPS (RTX 3050) |
| Congestion detection, fixed camera ROI | **Failed** — stayed `NORMAL` for all 300 frames despite a visibly congested scene (static occupancy peaked at 0.296) |
| Congestion detection, `uav_motion` + GMC | **Fixed** — correctly transitioned `NORMAL`→`CONGESTED` at frame 51, with 0 GMC lock failures across all 300 frames |

**Evidence.** `experiments/uav_pipeline_e2e_v1_20260818/run.json`; GMC
transform direction verified against a synthetic shift in
`tests/test_motion.py`; full writeup:
[benchmark protocol](docs/benchmark_protocol.md#uav-moving-camera-analytics-gmc).

**Caveat.** GMC (`cv2.findTransformECC`) is 2D image-plane motion
compensation only — not GPS/BEV georeferencing — and can lose lock on hard
scene cuts, fast motion, or low-texture frames; `gmc_consecutive_failures_at_end`
in `summary.json` must be checked per run, not assumed zero. This fix is
verified on one real clip, not a benchmark across multiple UAV sources.

## Known limitations

- Source videos collected from the web have incomplete provenance and cannot be
  assumed redistributable.
- The dataset remains small and class-imbalanced, especially for trucks, buses,
  and test pedestrians; broader geographic, weather, night, altitude, and
  camera-motion coverage is still required.
- All results are reported on a single fixed source-disjoint split (11 total
  source videos). With this few sources, which sources happen to land in
  test measurably changes results (see the Dataset section's source
  composition note). Cross-source validation (e.g. leave-one-source-out)
  would quantify that variance but was deliberately not run: at ~2-3h per
  training run, evaluating it across even one architecture is many GPU-hours
  on a single RTX 3050, and the specific cause of this split's variance
  (test happens to hold both aerial-drone sources) is already known and
  disclosed rather than averaged over. Treat every locked-test number in
  this readme as one sample from an unmeasured distribution, not a precise
  population estimate.
- The provenance-controlled tracking result is an integration baseline on
  VisDrone, not a Vietnam-domain or official VisDrone benchmark.
- Traffic speed requires camera calibration or a documented approximation.
  Bbox-union occupancy is image-plane box coverage, not physical road
  occupancy: boxes include background and there is no segmentation, BEV
  transform, or camera calibration.
- Line-crossing counts depend on stable ByteTrack identities. Occlusion-driven
  ID switches or fragmentation can cause duplicate or missed counts, and the
  error rate has not yet been measured on the two demo videos.
- VLM/LLM quality, hallucination rate, and quantization effects are not yet
  measured. `validate_grounding_policy` does not actually show clip frames to
  the VLM despite the policy name implying it would.
- Global motion compensation (`analytics.gmc_enabled`) is 2D image-plane
  alignment only, not BEV or GPS/IMU-based, and can lose lock on hard cuts,
  fast pans, or low-texture frames.
- Congestion detection depends entirely on the detector resolving individual
  boxes (`bbox_union_occupancy`, ROI track count). Under severe occlusion --
  a tightly packed, stalled crowd -- detector recall collapses exactly when
  congestion is worst, a false-negative blind spot at the extreme end
  (observed directly: a real rush-hour clip with a gridlocked motorcycle
  mass the detector drew zero boxes over stayed at `DENSE` instead of
  reaching `CONGESTED`). Two automatic-trigger attempts using a
  detection-independent optical-flow + texture signal
  (`src/vn_traffic/analytics/stillness.py`) were built, wired into
  `CongestionStateMachine`, and tested on the real motivating clip, and
  both were root-caused as not working: the underlying feature (Laplacian
  texture) cannot distinguish "packed vehicles" from "any other
  static, detailed surface" (buildings, signage), so no threshold or ROI
  choice fixes it -- confirmed by testing and rejecting two further
  hypotheses (an activity mask, a road-only ROI) on real data. What does
  work, real-pipeline-validated: a **visual heatmap**
  (`stillness_heatmap.enabled`, decoupled from the state machine, using a
  frame-relative threshold instead of a fixed one) that consistently tints
  the packed cluster across the clip for a human operator to see, even
  where the detector draws zero boxes. Automatic alerting for this failure
  mode remains unsolved -- see
  [benchmark protocol](docs/benchmark_protocol.md#detection-independent-stillness-signal-prototype)
  for the full negative-result trail and untried next candidates.
- The dashboard's live-frame write (`latest_frame.jpg`) can fail on Windows
  due to transient file locks (observed `PermissionError: [WinError 5]` from
  Defender/OneDrive scanning during a real run); this is handled as
  non-fatal, but the dashboard can occasionally show a stale frame.
- No model has yet been benchmarked on a physical edge NPU in this project.

## Roadmap

Current delivery priority is the deterministic CV product. Reasoning work is
limited to pretrained-model integration and demo quality; VLM/LLM fine-tuning,
all quantization work, and physical deployment are explicitly deferred.

- [x] Audit the legacy dataset and identify cross-split leakage.
- [x] Build source-grouped train/calibration/validation/test splits.
- [x] Lock test content by image and label hashes.
- [x] Add reproducible detector training and smoke validation.
- [x] Complete v5 fine-tuning and validation-based checkpoint selection.
- [x] Freeze the v5 detector and run its one-time locked-test evaluation.
- [x] Compare standard, high-resolution, SAHI, and hybrid inference on VisDrone-DET.
- [x] Feed the selected full-frame detector into ByteTrack once per source frame and compare 640 versus 1280.
- [x] Freeze and benchmark the complete CV pipeline end to end (with UAV alert-transfer failure documented).
- [x] Repair and validate sequence-level class-aware tracking evaluation.
- [x] Derive frame-count and image-space line-crossing ground truth from VisDrone-MOT trajectories and measure error.
- [x] Implement deterministic analytics and event schema with synthetic tests.
- [x] Complete initial ROI, counting-line, and congestion acceptance on two demo videos.
- [x] Add and synthetic-test a narrowly defined prolonged-stop alert with speed hysteresis and gap reset.
- [x] Add deterministic event keyframe/clip evidence selection.
- [x] Remove codec-dependent random seeking and add evidence provenance hashes.
- [x] Freeze VLM/LLM evaluation inputs and add JSON/prompt contract v1.
- [x] Add two-reviewer annotation templates, validation, and adjudication queue tooling.
- [x] Complete two independent reviewer annotation sets for reasoning evaluation v1.
- [ ] Resolve or formally defer the reasoning adjudication queue; it does not block CV delivery.
- [x] Freeze a separate run16 development set and record initial RTX model candidates.
- [x] Pin, hash, and smoke-test the Qwen3-VL-2B FP16 development adapter.
- [x] Demo existing pretrained VLM and LLM without tuning (functional smoke; quality not established).
- [x] Diagnose the UAV camera-motion ROI failure and implement global motion compensation (`analytics.mode: uav_motion`, `gmc_enabled`), verified on 300 real frames with zero GMC failures and a correct NORMAL→CONGESTED transition.
- [x] Fix the VLM/LLM prompt-copying bug (v1 to v3): eliminated copyable example sentences from both the VLM and LLM prompts, verified with grounded, distinct, analytics-consistent output on two real clips.
- [x] Add a Streamlit dashboard over pipeline run output (headless boot verified; live browser auto-refresh not yet human-confirmed).
- [x] Run a bounded end-to-end UAV benchmark on the RTX host.
- [x] Integrate TrackEval for HOTA/DetA/AssA and decompose the tracking bottleneck (detection-limited, not association-limited).
- [x] Test a BoT-SORT/ReID tracking ablation against the ByteTrack baseline (algorithm switch helped, ReID itself did not).
- [x] Test an NWD bbox-loss ablation against the detector's small-object generalization gap (rejected: worse than baseline CIoU on every locked-test metric and class).
- [x] Test a P2 detection-head architecture ablation (adds a stride-4 output) against the same gap (rejected: worse than baseline and worse than NWD on every locked-test metric; training required recovering from a CUDA OOM, a BatchNorm corruption at batch=1, and an Ultralytics `resume=True` bug -- see [benchmark protocol](docs/benchmark_protocol.md#p2-detection-head-ablation-rejected)).
- [ ] Planned: test copy-paste augmentation of small objects (oversampling the rare <16px train boxes) against the same gap -- addresses the underlying data imbalance directly (4.7% of train boxes are under 16px versus 48.8% of test boxes; see the Dataset section) rather than the loss or architecture. Now the leading candidate: both the loss-side (NWD) and architecture-side (P2) fixes have failed.
- [x] Detection-independent "stalled and dense" signal for severe-occlusion jams (`src/vn_traffic/analytics/stillness.py`): built, unit-tested, and real-pipeline-validated. The automatic `CONGESTED`-trigger path (wired into the state machine, not gated by detected speed) is a **negative result, root-caused**: Laplacian texture cannot distinguish packed vehicles from any other static detailed surface, so no threshold/ROI fix works (two further hypotheses tested and rejected on real data -- see benchmark protocol). The **visual heatmap** path (`stillness_heatmap.enabled`, decoupled from the state machine) **works**: real-pipeline-validated to consistently tint the packed cluster the detector misses, across the whole clip.
- [ ] Open: an automatic trigger for this failure mode needs a different feature, not a retuned threshold -- candidates (neither implemented): low-confidence pre-NMS detector proposals as a coarse density prior, or a texture filter band-passed to vehicle/head size. Also open: per-lane/multi-region ROI decomposition (Stage 3) and GMC ego-motion compensation so the heatmap also works in `uav_motion` mode.
- [ ] Deferred: evaluate a newer Ultralytics architecture generation (e.g. YOLO26) as a new baseline; not assumed better or worse than YOLOv8 until measured.
- [ ] Deferred: export and benchmark detector FP16/INT8 candidates.
- [ ] Deferred: quantize and benchmark the selected VLM and LLM.
- [ ] Deferred beyond current goal: validate an appropriate physical edge/NPU target.

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
