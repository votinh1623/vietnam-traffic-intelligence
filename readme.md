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

**Evidence.** `experiments/yolov8s_v5_locked_test_20260818/run.json`,
`experiments/visdrone_det_small_object_v1_20260818/run.json`.

**Caveat.** The validation-to-test gap is real and driven by an
object-scale/source shift (82.9%-100% of locked-test boxes cover under 0.1%
of the image, by class, versus 1.3%-70.0% on validation), not an evaluation
bug. V5 is a functional prototype, not a production-general detector. The
historical `vietnam_dataset_v2` result (`mAP50=0.745`) is excluded here
because it is invalid for scientific comparison (split leakage). Full
per-class diagnosis: [benchmark protocol](docs/benchmark_protocol.md).

### Tracking

**Setup.** ByteTrack over the selected VisDrone checkpoint, all 2,846 frames
across 7 VisDrone2019-MOT-val sequences, class-aware IoU matching at 0.5.

| Metric | Value |
|---|---:|
| IDF1 | 0.309 |
| MOTA | 0.020 |
| MOTP distance | 0.289 |
| ID switches | 462 |
| Fragmentations | 1,491 |
| HOTA / DetA / AssA | Pending TrackEval integration |

Resolution comparison (640 vs. 1280, vehicle classes only, same sequences):

| Mode | IDF1 | MOTA | Precision | Recall | ID switches |
|---|---:|---:|---:|---:|---:|
| Standard 640 | 0.473 | **0.215** | **0.663** | 0.449 | **411** |
| Standard 1280 | **0.481** | 0.132 | 0.578 | **0.521** | 568 |

**Evidence.** `experiments/tracking_visdrone_mot_val_v1_20260818/run.json`,
`experiments/tracking_visdrone_mot_resolution_v1_20260818/run.json`.

**Caveat.** This is a provenance-controlled integration baseline, not an
official VisDrone benchmark (no ignore-region handling) or Vietnam-domain
evidence. Low recall and 462 ID switches remain a material limitation for
downstream counting. A candidate with aligned 0.4 track/new thresholds was
tested and rejected (worse IDF1, MOTA, and +257 ID switches). Full numbers:
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
