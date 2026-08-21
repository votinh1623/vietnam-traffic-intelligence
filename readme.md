# Vietnam Traffic Intelligence

Leakage-controlled traffic detection, tracking, counting, alerting, and
multimodal reporting for UAV traffic video, developed and measured on an
NVIDIA RTX 3050 Laptop GPU (6 GB VRAM). Quantization and physical edge/NPU
deployment are explicitly deferred until the current research goal is
complete.

See [docs/quickstart.md](docs/quickstart.md) for installation and CLI usage,
and [docs/literature_review.md](docs/literature_review.md) for the papers
behind the algorithms and problems below.

---

## Why this project

Vietnamese road scenes are dense, dominated by small motorcycles and
pedestrians, and frequently affected by occlusion and camera motion. A useful
system must do more than draw boxes: it must preserve identities, aggregate
traffic state, explain noteworthy events, and report the accuracy and runtime
cost of every optimization honestly. **Core focus (2026-08-21 onward):
small-object detection and tracking accuracy**, developed and evaluated on
VisDrone rather than the project's own small Vietnam dataset -- see
[Dataset](#dataset) for why.

## Research objectives

The current goal is complete only when the project demonstrates and evaluates
all of the following, without relying on quantization or physical deployment:

1. detect and count vehicles from UAV video;
2. improve counting reliability with tracking and explicit handling of
   detection/tracking errors;
3. use pretrained VLM/LLM models to generate a traffic description;
4. emit alerts for high density or a narrowly defined abnormal event; and
5. evaluate the system on real UAV data, on VisDrone as the primary dataset.

| Goal | Maps to objective(s) | Required evidence for completion |
|---|---|---|
| Detection | 1 | Standard-versus-sliced inference on VisDrone-DET with overall, per-class, object-scale, and latency metrics |
| Tracking | 2 | Class-aware IDF1, MOTA, MOTP distance, ID switches, and fragmentations on VisDrone-MOT |
| Counting | 1, 2 | Ground-truth trajectory-derived line-crossing counts and error metrics; comparison against the selected tracker output |
| Alerts | 4 | Deterministic high-density alert plus explicitly configured wrong-way or prolonged-stop event, with synthetic tests and video evidence |
| VLM/LLM description | 3 | At least one end-to-end report from pretrained models with structured-event numbers kept separate from visual claims |
| UAV system evaluation | 5 | Reproducible detector, tracker, counting, alert, and end-to-end results with model/config/data hashes |

SAHI is evaluated first as an inference-only small-object method on VisDrone;
it is not assumed to improve tracking until measured.

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

**Primary dataset: VisDrone2019-DET/MOT.** Detector and tracker development,
selection, and gating now run entirely on VisDrone (train 6,471 / val 548
images for DET; MOT-val for tracking). This supersedes the project's own
small Vietnam v5 dataset as the active benchmark -- see
[Field validation](#field-validation-vietnam-clips-historical) for what
Vietnam v5 is retained for.

**Locked test: VisDrone2019-DET-test-dev (1,610 images, public GT), placed
and locked 2026-08-21.** `test-challenge` is not usable locally (ground
truth withheld). The first read on it is already a real finding, not a
formality: every prior decision (checkpoint selection, inference-mode
selection, the highres-pilot gate, tracker/ReID comparisons) had repeatedly
used the same 548-image val set, and the highres pilot's AP-small gain does
**not** clearly replicate on test-dev -- see
[Detection](#detection). Confirms the risk was real, not hypothetical; from
here on, val-based selection-era results and test-dev reads are both
labeled as such.

## Evaluation policy

- Select checkpoints and thresholds on validation.
- Never use VisDrone2019-DET-test-dev for model, tracker, prompt, or
  backend selection -- only for confirmatory reads reported as-is. The
  highres-pilot promotion (val-based, before test-dev was locked) is not
  retroactively reversed by a test-dev read; future levers must be gated
  against test-dev directly instead.
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
section states outcomes only. All results below are VisDrone-native (see
[Dataset](#dataset) for the val-vs-locked-test caveat that currently applies
to all of them).

### Detection

YOLOv8s initialized from COCO, fine-tuned on VisDrone2019-DET
(`mAP50=0.389` @ epoch 74). Why YOLOv8 over newer Ultralytics generations:
[benchmark protocol](docs/benchmark_protocol.md#why-yolov8).

Small-object inference-mode selection (548 VisDrone-DET val images, COCO-style AP):

| Mode | AP | AP-small | p50 latency (ms/img) | Decision |
|---|---:|---:|---:|---|
| Standard 640 | 0.212 | 0.118 | 23.7 | Reference |
| **Standard 1280** | **0.264** | **0.194** | 130.8 | **Selected mode** |
| SAHI 640 tiles | 0.193 | 0.142 | 282.0 | Rejected |
| Hybrid full-frame + tiles | 0.177 | 0.117 | 200.1 | Rejected |

**Checkpoint promoted 2026-08-21 (val-based decision).** The mode-selection
checkpoint above was trained only at 640 despite inferring at 1280 -- a
train/infer resolution mismatch. A 5-epoch native-1280 continuation, gated
in advance at AP-small +0.010 absolute / overall AP drop <=0.005, passed on
val with margin: AP 0.264->0.296 (+0.0325), AP-small 0.194->0.216 (+0.0223)
-- and propagated into a real tracking gain (see [Tracking](#tracking)).

**Locked test-dev first read (2026-08-21): the AP-small gain does not
clearly replicate.**

| Split | Baseline AP / AP-small | Pilot AP / AP-small | Delta |
|---|---:|---:|---:|
| val (selection-era) | 0.264 / 0.194 | 0.296 / 0.216 | AP +0.0325, AP-small +0.0223 |
| **test-dev (locked, first read)** | 0.215 / 0.138 | 0.229 / 0.141 | AP +0.0144, AP-small **+0.0027** |

The test-dev AP-small delta is far under the +0.010 gate that decided the
promotion -- consistent with the val-selection risk flagged in
[Dataset](#dataset): the same 548 val images were reused across many
experiments before this. The promotion is **not reversed** by a single
test-dev read (that would make test-dev another selection surface), but its
practical small-object benefit should be treated as **unconfirmed**, not
established, until a future lever is tested directly against test-dev.
Full writeup:
[benchmark protocol](docs/benchmark_protocol.md#visdrone-highres-fine-tune-pilot-and-checkpoint-promotion).

**Evidence.** `experiments/visdrone_det_small_object_v1_20260818/run.json`,
`experiments/visdrone_highres_pilot_and_reid_results_20260821/run.json`,
`experiments/visdrone_testdev_locked_first_read_20260821/run.json`.

### Tracking

Resolution comparison (640 vs. 1280, VisDrone ten-class checkpoint filtered
to eight vehicle classes, confidence 0.1, ByteTrack, all 2,846 frames across
7 VisDrone2019-MOT-val sequences):

| Mode | IDF1 | MOTA | Precision | Recall | ID switches |
|---|---:|---:|---:|---:|---:|
| Standard 640 | 0.473 | **0.215** | **0.663** | 0.449 | **411** |
| Standard 1280 | **0.481** | 0.132 | 0.578 | **0.521** | 568 |

Same protocol, swapping in the promoted highres-pilot checkpoint (see
[Detection](#detection)):

| Metric | 640-trained | Highres pilot | Delta |
|---|---:|---:|---:|
| IDF1 | 0.481 | 0.504 | +0.023 |
| MOTA | 0.132 | 0.221 | +0.089 |
| Precision | 0.578 | 0.642 | +0.064 |
| ID switches | 568 | 454 | -114 |

A real, propagating detection-quality gain, recall roughly unchanged (not
traded away). A parallel ReID ablation (real pretrained embedding
`yolo26n-reid.onnx` vs. `model:auto` feature reuse) showed **no
improvement**, reconfirming detection recall -- not association -- as the
bottleneck. Full detail:
[benchmark protocol](docs/benchmark_protocol.md#bot-sort-and-reid-ablation).

**Evidence.** `experiments/tracking_visdrone_mot_resolution_v1_20260818/run.json`,
`experiments/visdrone_highres_pilot_and_reid_results_20260821/run.json`.

**Caveat.** Not an official VisDrone benchmark (no ignore-region handling).
`scripts/evaluate_hota.py` had a bug silently dropping zero-prediction
frames instead of scoring them as false negatives (32-33 of 2,846 frames);
fixed, moved every HOTA number by <=0.002 in the field-validation table
below, no conclusion changed. DetA stayed nearly flat across tracker
algorithms while AssA moved with tracker choice -- see
[benchmark protocol](docs/benchmark_protocol.md#bot-sort-and-reid-ablation).

### Counting

Production analytics state machine against native VisDrone MOT ground-truth
trajectories, 2,382 frames across 6 traffic sequences, three frozen
horizontal counting lines, VisDrone-native checkpoint.

| Tracking profile | Frame-count macro MAE (veh/frame) | Frame-count micro WAPE | Crossing WAPE |
|---|---:|---:|---:|
| Standard 640 | 10.45 | 0.372 | 0.593 |
| **Standard 1280** | **9.80** | **0.319** | **0.560** |

**Evidence.** `experiments/counting_visdrone_mot_v1_20260818/run.json`.

**Caveat.** Crossing WAPE of 0.560 demonstrates measurable counting, not
production-grade accuracy -- no stabilization or BEV transform is applied,
so image-space crossings measure agreement with GT trajectories, not
calibrated physical flow.

### VLM/LLM description

Two-stage pretrained contract, no fine-tuning: Qwen3-VL-2B-Instruct (visual
assessment) feeds Qwen3-0.6B (Vietnamese report). Numeric facts are
assembled from the deterministic event, not generated, and validated
automatically against it. Demonstrated on real field-validation clips (see
below) -- this objective is about the reasoning contract, not detection
accuracy, so it is not tied to the VisDrone-vs-Vietnam dataset question.

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
despite the name -- an open gap.

### UAV system evaluation

Full pipeline (VisDrone-native detector + ByteTrack + analytics + evidence
export) run end to end on 300 real 1080p UAV frames. Re-run 2026-08-21 after
promoting the highres-pilot checkpoint; original 2026-08-18 numbers kept for
comparison.

| Metric | 2026-08-18 (640-trained) | 2026-08-21 (highres pilot) |
|---|---:|---:|
| End-to-end throughput | 3.70 FPS | 3.40-3.46 FPS (normal run-to-run variance, not a new bottleneck) |
| Congestion, fixed camera ROI | **Failed** -- stayed `NORMAL` for all 300 frames | **Still fails the same way** -- see caveat |
| Congestion, `uav_motion` mode | **Fixed** -- `NORMAL`->`CONGESTED` at frame ~64-65 | **Reconfirmed** -- same transition frame |

**Evidence.** `experiments/uav_pipeline_e2e_v1_20260818/run.json`,
`experiments/visdrone_highres_pilot_and_reid_results_20260821/run.json`.
Full GMC A/B and re-confirmation detail:
[benchmark protocol](docs/benchmark_protocol.md#uav-moving-camera-analytics-gmc).

**Caveat.** The `NORMAL`->`CONGESTED` fix is `analytics.mode: uav_motion`'s
count-alone trigger, not GMC -- an A/B test shows the same transition frame
with or without GMC. The `fixed_camera` failure is a mode/threshold
limitation on this clip, not a detection-recall gap, so it did not change
when the detector improved. This remains one real clip, not a benchmark
across multiple UAV sources.

### Field validation (Vietnam clips, historical)

Before 2026-08-21, the project's own small Vietnam dataset (leakage-fixed as
**Vietnam v5**: source-disjoint splits, content-addressed test lock -- full
history in [the dataset protocol](docs/dataset_protocol.md)) was the primary
benchmark. It is now retained only as a practical, qualitative field check
-- real clips the pipeline should handle reasonably, not a formal
train/val/test evaluation -- because VisDrone is a larger, more diverse, and
now-primary benchmark for the small-object-accuracy goal (see
[Dataset](#dataset)).

| Check | Result |
|---|---|
| Detector, Vietnam v5 locked test (176 images) | mAP50-95=0.062 (vs. 0.344 on its own validation split) -- a real object-scale/source-shift gap, not an evaluation bug |
| NWD bbox-loss ablation for the same gap | **Rejected** -- worse than baseline on every metric |
| P2 detection-head ablation for the same gap | **Rejected** -- worse than baseline and NWD |
| Tracking (Vietnam v5 5-class checkpoint, ByteTrack/BoT-SORT/ReID) | HOTA 0.288/0.322/0.324; DetA flat, AssA moves with tracker -- same detection-limited pattern as the VisDrone tracking result above |
| Alerts, 2 demo clips (`traffic_jam.mp4`, `traffic_normal.mp4`) | Congestion state machine and prolonged-stop alert accepted qualitatively; not precision/recall evidence |

Both loss-side (NWD) and architecture-side (P2) small-object fixes were
tried on Vietnam v5 and failed; this is consistent with the gap being a
training-data scarcity problem (only 12 total source videos) rather than
something either change alone corrects -- see the
[dataset protocol](docs/dataset_protocol.md#source-composition-and-the-object-scale-gap).
This is exactly the class of experiment now redirected to VisDrone instead.

**Evidence.** `experiments/yolov8s_v5_locked_test_20260818/run.json`,
`experiments/yolov8s_v5_seed0_nwd_20260819T094236/run.json`,
`experiments/yolov8s_v5_seed0_p2_continued_20260819T152917/run.json`,
`experiments/tracking_visdrone_mot_val_v1_20260818/run.json`,
`experiments/tracking_hota_corrected_20260820/run.json`,
`experiments/alerts_acceptance_v1_20260818/run.json`.

## Known limitations

- Source videos collected from the web have incomplete provenance and cannot
  be assumed redistributable.
- The promoted detector checkpoint's AP-small gain was gated and passed on
  val (+0.0223) but does **not** clearly replicate on the now-locked
  VisDrone2019-DET-test-dev (+0.0027, under the +0.010 gate) -- see
  [Detection](#detection). A ground-truth distribution diagnostic
  (`scripts/analyze_visdrone_split_gap.py`) rules out class-averaging as
  the cause (the gap persists restricted to vehicle classes only) but does
  not isolate a single fix-able root cause the way the Vietnam v5 gap did
  -- the baseline checkpoint's own vehicle AP-small also drops 23%
  relative from val to test-dev, so test-dev looks intrinsically harder
  in general, not specifically resistant to this fix. Full diagnostic:
  [benchmark protocol](docs/benchmark_protocol.md#why-the-gain-didnt-replicate-val-vs-test-dev-diagnostic).
  Treat the checkpoint's small-object benefit as unconfirmed until
  retested with a lever developed without repeated val checks and gated
  against test-dev exactly once.
- Tracking, counting, and ReID comparisons still have no equivalent locked
  test (only VisDrone-MOT-val exists) -- the same inflation risk applies
  there and is unmeasured.
- The tracking result is an integration baseline on VisDrone, not an
  official VisDrone benchmark (no ignore-region handling).
- Traffic speed requires camera calibration or a documented approximation.
  Bbox-union occupancy is image-plane box coverage, not physical road
  occupancy.
- Line-crossing counts depend on stable ByteTrack identities; the
  occlusion-driven error rate has not been measured.
- VLM/LLM quality, hallucination rate, and quantization effects are not yet
  measured.
- Global motion compensation is 2D image-plane alignment only, not BEV or
  GPS/IMU-based, and can lose lock under hard cuts/fast pans/low-texture
  frames -- check `gmc_total_failures` (run-wide), not just
  `gmc_consecutive_failures_at_end` (end-of-run streak only). See
  [UAV system evaluation](#uav-system-evaluation) for the GMC-vs-`uav_motion`
  distinction.
- Congestion detection depends on the detector resolving individual boxes.
  Under severe occlusion, detector recall collapses exactly when congestion
  is worst. A detection-independent stillness signal was tested as an
  automatic trigger; **rejected and root-caused**. A **visual heatmap**
  variant **works**. Full trail:
  [benchmark protocol](docs/benchmark_protocol.md#detection-independent-stillness-signal-prototype).
- The dashboard's live-frame write can fail on Windows due to transient file
  locks (handled as non-fatal).
- No model has yet been benchmarked on a physical edge NPU.

## Roadmap

Priority: small-object detection and tracking accuracy on VisDrone. Full
history of completed work, experiments (including rejected ones), and minor
fixes: [docs/history.md](docs/history.md).

1. **Find a small-object lever that survives test-dev.** VisDrone2019-DET-test-dev
   is locked (2026-08-21); the promoted checkpoint's AP-small gain did not
   clearly replicate on it (see [Detection](#detection)). Retest candidates
   (e.g. scale-aware copy-paste, more highres-pilot epochs) gated against
   test-dev directly, not val.
2. **Lock an equivalent test for tracking/ReID** -- only VisDrone-MOT-val
   exists today, same inflation risk as detection had.
3. **Close the congestion product gap**: the confidence fix still leaves
   the motivating clip mostly `NORMAL`.
4. TVLR Stage C stays paused; VLM/LLM fine-tuning, quantization, and
   physical edge/NPU deployment stay deferred.

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
