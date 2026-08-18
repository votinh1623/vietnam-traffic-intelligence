# Vietnam Traffic Intelligence

Leakage-controlled traffic detection, tracking, counting, alerting, and
multimodal reporting for UAV traffic video.

The current system fine-tunes YOLOv8 on Vietnamese traffic scenes and provides
the foundation for ByteTrack-based tracking, structured traffic analytics, and
event-driven VLM/LLM interpretation. Development and measurement currently run
on an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM. Quantization and physical
edge/NPU deployment are explicitly deferred until the current research goal is
complete.

---

## Why this project

Vietnamese road scenes are dense, dominated by small motorcycles and
pedestrians, and frequently affected by occlusion and camera motion. A useful
system must do more than draw boxes: it must preserve identities, aggregate
traffic state, explain noteworthy events, and report the accuracy and runtime
cost of every optimization honestly.

This repository therefore separates four current concerns:

1. dataset integrity and leakage-controlled evaluation;
2. object detection and multi-object tracking;
3. traffic analytics and multimodal interpretation;
4. source-disjoint evaluation on real UAV datasets such as VisDrone.

## What it does

| Capability | Current implementation | Status |
|---|---|---|
| Image and video detection | YOLOv8, five Vietnamese traffic classes | Implemented |
| Multi-object tracking | Ultralytics ByteTrack integration and custom tracker configuration | Implemented; provenance-controlled v5 integration benchmark complete |
| Structured traffic analytics | Bbox-union ROI coverage, trajectories, directional counts, and congestion events | Implemented; initial two-video acceptance passed |
| UAV moving-camera analytics | `analytics.mode: uav_motion`, optional ECC-based GMC re-projecting the ROI/counting-line into every frame | Implemented; real-UAV-clip run corrected a NORMAL-stuck-despite-visible-congestion failure, direction of the ECC transform verified against a known synthetic shift |
| Event evidence selection | Raw keyframes for deterministic events and clips for congestion transitions | Implemented; two-video acceptance passed |
| Local monitoring dashboard | Streamlit app reading a `run_pipeline.py` output directory: live annotated frame (`latest_frame.jpg`, atomic per-frame write), current state, timeline, recent events, finished video | Implemented; headless boot, real-run data load, and per-frame atomic JPEG write/decode all verified, live browser auto-refresh not yet visually confirmed by a human |
| VLM scene understanding | Representative keyframe per event; Qwen3-VL-2B FP16 | Implemented; pretrained smoke passed on real jam/normal clips, task quality not formally measured |
| LLM reasoning and reports | Event-driven Vietnamese summaries assembled from structured analytics plus VLM evidence | Implemented; pretrained smoke passed (Qwen3-0.6B fallback), task quality not formally measured |
| Detector quantization | FP16/INT8 export and accuracy-latency evaluation | Deferred beyond current goal |
| VLM/LLM quantization | Backend-specific FP16/INT8 or weight-only evaluation | Deferred beyond current goal |
| Physical edge/NPU execution | Target-specific validation | Deferred beyond current goal |

See [the multimodel architecture](docs/multimodel_architecture.md) for component
boundaries and the intended event-driven VLM/LLM path.

## Current research goal

The current goal is complete only when the project demonstrates and evaluates
all of the following without relying on quantization or physical deployment:

1. detect and count vehicles from UAV video;
2. improve counting reliability with tracking and explicit handling of
   detection/tracking errors;
3. use pretrained VLM/LLM models to generate a traffic description;
4. emit alerts for high density or a narrowly defined abnormal event; and
5. evaluate the system on real UAV data, primarily VisDrone.

| Goal | Required evidence for completion |
|---|---|
| Small-object detection | Standard-versus-sliced inference on VisDrone-DET with overall, per-class, object-scale, and latency metrics |
| Tracking | Class-aware IDF1, MOTA, MOTP distance, ID switches, and fragmentations on VisDrone-MOT |
| Counting | Ground-truth trajectory-derived line-crossing counts and error metrics; comparison against the selected tracker output |
| Alerts | Deterministic high-density alert plus explicitly configured wrong-way or prolonged-stop event, with synthetic tests and video evidence |
| VLM/LLM description | At least one end-to-end report from pretrained models with structured-event numbers kept separate from visual claims |
| UAV system evaluation | Reproducible detector, tracker, counting, alert, and end-to-end results with model/config/data hashes |

SAHI is evaluated first as an inference-only small-object method on VisDrone;
it is not assumed to improve tracking until merged full-frame detections have
been passed once per source frame to ByteTrack and measured. The consumed
Vietnam v5 locked test remains final and is not used to select slicing
parameters.

## System architecture

The intended data flow is:

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
memory use, and hallucination surface.

## Research status

| Area | Status | Evidence |
|---|---|---|
| Audited Python/CUDA environment | Complete | [Environment](docs/environment.md) |
| Legacy Vietnam v2 leakage audit | Complete; invalid for scientific claims | [Dataset protocol](docs/dataset_protocol.md) |
| Source-grouped Vietnam v5 dataset | Complete | `configs/datasets/vietnam_v5.yaml` |
| Content-addressed locked test | Complete, 176 images | `manifests/datasets/vietnam_v5/test_lock.json` |
| YOLOv8s v5 smoke run | Complete | `experiments/yolov8s_v5_seed0_smoke_20260817T100534/run.json` |
| YOLOv8s v5 full fine-tuning | Complete, 30 epochs | `experiments/yolov8s_v5_seed0_20260817T100644/run.json` |
| YOLOv8s v5 locked-test evaluation | Complete once; no further test-driven tuning | `experiments/yolov8s_v5_locked_test_20260818/run.json` |
| Offline-video CLI and artifact contract | Integrated with v5 checkpoint | [Output schema](docs/output_schema.md) |
| Deterministic analytics state machine | Synthetic tests and bbox-union two-video acceptance passed | [Output schema](docs/output_schema.md) |
| Tracking evaluator | IoU association and OVERALL MOTP aggregation repaired; 7-sequence v5 integration benchmark complete | [Benchmark protocol](docs/benchmark_protocol.md) |
| Export and quantization benchmark | Deferred beyond current goal | [Benchmark protocol](docs/benchmark_protocol.md) |
| Event evidence selector | Sequential no-seek exporter implemented; two-video acceptance passed | [Multimodel architecture](docs/multimodel_architecture.md) |
| VLM/LLM reasoning contract | Inputs locked; both reviews complete; one disagreement and final adjudication pending | [Reasoning protocol](docs/reasoning_protocol.md) |
| VLM/LLM model inference | Sequential pretrained demo passed on real `traffic_jam.mp4`/`traffic_normal.mp4` runs after the v3 prompt fix (grounded, non-generic, matching real analytics on both clips); hallucination rate and output quality remain formally unmeasured | [Reasoning protocol](docs/reasoning_protocol.md) |

The detector locked test was consumed once after validation-only checkpoint
selection. Its result is final for v5 and must not be used for further model or
threshold selection.

## Dataset governance

The original `vietnam_dataset_v2` split contained severe temporal and source
leakage: all 12 source videos appeared across train, validation, and test. Its
metrics are preserved as historical evidence only.

Vietnam v5 is materialized non-destructively with source-disjoint splits,
polygon-to-box conversion, deterministic removal of 53 exact duplicate boxes,
and a content-addressed test lock.

| Split | Images | Bus | Car | Motorcycle | Pedestrian | Truck | Total boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 819 | 1,559 | 12,439 | 29,005 | 3,265 | 2,781 | 49,049 |
| Calibration | 111 | 158 | 742 | 4,360 | 1,259 | 23 | 6,542 |
| Validation | 108 | 75 | 872 | 2,554 | 2,462 | 60 | 6,023 |
| Locked test | 176 | 622 | 6,038 | 4,486 | 170 | 327 | 11,643 |
| **Total** | **1,214** | **2,414** | **20,091** | **40,405** | **7,156** | **3,191** | **73,257** |

Important limitations:

- calibration contains only 23 truck boxes;
- validation contains 75 bus and 60 truck boxes;
- locked test contains only 170 pedestrian boxes;
- four source-unknown frames and 74 conflicting duplicate-frame annotation
  groups were excluded;
- four short sources are renamed YouTube videos; the DJI source is separate;
- the visual near-overlap audit found no candidate overlap between those
  sources at a dHash distance threshold of 12/256, but this does not prove
  ownership or redistribution rights.

The raw videos and generated datasets are intentionally excluded from Git.
Users are responsible for the rights and licenses of their own source media.

## Detector results

YOLOv8s was initialized from COCO, fine-tuned on VisDrone2019-DET
(`mAP50=0.389` at epoch 74), then fine-tuned again on the leakage-controlled
Vietnam v5 dataset with full weights (`freeze=0`, 30 epochs, input 1280;
epoch 29 selected on validation only). Standard 1280 inference was selected
over SAHI/hybrid tiling for the downstream tracking pipeline (best AP,
AP-small, and AR among four compared modes, at about 2.2x SAHI's speed). The
historical `vietnam_dataset_v2` result (`mAP50=0.745`) is invalid for
scientific comparison because that split leaked sources across train,
validation, and test.

| Split | Precision | Recall | mAP50 | mAP50-95 | Use |
|---|---:|---:|---:|---:|---|
| Validation | 0.762 | 0.504 | 0.600 | 0.344 | Checkpoint selection |
| Locked test | 0.215 | 0.287 | 0.148 | 0.062 | Final reporting only |

The validation-to-test gap is real and driven by an object-scale/source
shift (locked-test boxes are far smaller), not an evaluation bug; v5 is a
functional prototype, not a production-general detector. Full training
hyperparameters, the small-object inference comparison, and the per-class
generalization-gap diagnosis are in
[the benchmark protocol](docs/benchmark_protocol.md); checkpoint and dataset
hashes are in `experiments/yolov8s_v5_locked_test_20260818/run.json`.

## Tracking and analytics

ByteTrack is integrated through `src/vn_traffic/perception.py` and
`bytetrack_custom.yaml`. The local motmetrics evaluator originally inverted
an already distance-form IoU matrix and produced invalid associations
(historical `tracking_metrics_*.csv` outputs were removed); the repaired
evaluator uses `1-IoU` directly, includes prediction-only frames, and
aggregates a combined OVERALL accumulator, covered by synthetic regression
tests. The provenance-controlled integration baseline (VisDrone-MOT-val, 7
sequences / 2,846 frames, `imgsz=1280`) is not an official VisDrone
benchmark or Vietnam-domain evidence; its low recall and 462 ID switches
show tracking remains a material limitation for counting.

A resolution comparison (640 vs. 1280, vehicle classes only) traded 0.085
precision and 0.082 MOTA for 0.071 recall and 0.009 IDF1, so resolution
selection was deferred to the downstream counting task instead of being
decided from identity metrics alone. There, standard 1280 reduced both
frame-count and line-crossing error against ground-truth VisDrone MOT
trajectories (micro WAPE 0.372 to 0.319; crossing WAPE 0.593 to 0.560) and
was selected — but crossing WAPE of 0.560 means this demonstrates
measurable counting, not production-grade accuracy, especially since these
UAV cameras move/zoom and no stabilization or BEV transform is applied. Full
numbers for both comparisons are in
[the benchmark protocol](docs/benchmark_protocol.md).

The deterministic analytics engine (`src/vn_traffic/analytics`) tracks
per-track trajectories, counts directional line crossings, measures unique
bbox-union ROI coverage, estimates centroid speed, and applies hysteresis
across `NORMAL`/`DENSE`/`CONGESTED`; a narrowly defined `prolonged_stop`
alert covers eligible tracks with synthetic-tested duration/release/gap
handling. Schema and artifact details are in
[the output schema](docs/output_schema.md). A clean two-clip acceptance run
separated `traffic_normal.mp4` (`NORMAL` throughout) from `traffic_jam.mp4`
(`NORMAL`→`CONGESTED` at 2.12 s), but thresholds were only tuned against
these two scenes and do not yet transfer to other cameras — see
[the benchmark protocol](docs/benchmark_protocol.md) for the full
acceptance numbers.

The end-to-end product-path benchmark (300 real 1080p UAV frames, selected
VisDrone 1280 profile, full pipeline including evidence export) ran at 3.70
FPS on the RTX 3050, but exposed the main unresolved failure at the time:
despite a visually congested scene, the moving/zooming camera kept the
fixed-ROI state machine `NORMAL` for all 300 frames (static occupancy peaked
at only 0.296). `analytics.mode: uav_motion` with optional
`gmc_enabled` (ECC-based global motion compensation, verified against a
synthetic shift in `tests/test_motion.py`) fixes this: re-run on the same
clip, it correctly transitioned `NORMAL`→`CONGESTED` at frame 51 with zero
GMC lock failures across all 300 frames. GMC remains 2D image-plane
compensation only (no BEV/GPS) and can lose lock under hard cuts or
low-texture frames. Full details are in
[the benchmark protocol](docs/benchmark_protocol.md#uav-moving-camera-analytics-gmc).

Stage 3 adds a deterministic evidence boundary (sequential, no-seek frame
decode; hashed keyframes and clips) before any VLM is loaded, documented in
[the output schema](docs/output_schema.md). Two acceptance runs validated
it end to end: `traffic_jam.mp4` (14/14 events, decoded-frame hashes, and
artifact hashes verified) and `traffic_normal.mp4` (146 events, 137/137
unique frame hashes verified). Reasoning evaluation v1 then freezes all 14
`traffic_jam.mp4` evidence records under lock SHA-256
`ecfd9a1e44ae1be4991f5e87dbf65d3ce9e42c4185a00db782e728258673d18b`; input
lock is complete, but human reference annotations are still pending, so no
reasoning-model quality result is claimed yet. See the
[reasoning protocol](docs/reasoning_protocol.md).

A significant early VLM/LLM bug is worth calling out here: every run before
the fix copied a literal example sentence out of the prompt instead of
describing the actual image, regardless of image content — including once
onto a factually wrong answer for a truck-dominated scene. The three-attempt
fix (replacing every copyable example sentence with a fill-in-the-brackets
template) is documented in
[the reasoning protocol](docs/reasoning_protocol.md#prompt-copying-bug-v1-to-v3),
along with a still-open gap where clip evidence is referenced but never
actually shown to the VLM.

| Metric | Current status |
|---|---|
| HOTA | Pending TrackEval integration |
| IDF1 | 0.309 on the provenance-controlled VisDrone integration baseline |
| MOTA | 0.020 on the provenance-controlled VisDrone integration baseline |
| ID switches | 462 across 7 sequences / 2,846 frames |
| Detector + tracker latency | Pending benchmark |

## Deployment readiness

Deployment readiness will be evaluated as a system-level trade-off, not merely
as model file conversion.

| Model stage | Candidate formats | Required evidence |
|---|---|---|
| Detector | PyTorch FP32/FP16, ONNX FP32/FP16/INT8, TensorRT | Accuracy, pre/infer/post latency, throughput, VRAM, size |
| VLM | PyTorch/Transformers FP16, INT8 or weight-only candidate | Task quality, latency, VRAM/RAM, size |
| LLM | FP16 baseline, INT8/INT4 candidate where supported | Report quality, tokens/s, first-token latency, memory, size |
| End-to-end pipeline | Host runtime, then target-specific edge runtime | Event latency, throughput, resource use, fallback behavior |

INT8 calibration must use the calibration split only. The locked test is not
used to select precision, thresholds, prompts, runtime backends, or tracker
parameters. ONNX Runtime, TensorRT, and physical NPU claims remain pending until
their backends and target hardware are installed and measured.

## Repository layout

```text
.
|-- configs/
|   |-- datasets/            # audited dataset metadata
|   |-- experiments/         # reproducible experiment definitions
|   |-- pipeline/            # offline-video runtime configuration
|   `-- reasoning/           # versioned VLM/LLM prompts
|-- docs/                    # protocols, architecture, and environment evidence
|-- experiments/             # lightweight run manifests and hashes
|-- manifests/
|   |-- datasets/            # leakage audits and detector test locks
|   `-- reasoning/           # content-addressed VLM/LLM input locks
|-- scripts/
|   |-- data/                # audit, materialization, overlap, and lock tools
|   |-- reasoning/           # content-addressed evidence-set tooling
|   |-- train/               # provenance-aware detector training
|   |-- detect.py            # image, directory, and video inference
|   |-- tracking_metrics.py  # tested MOT metric implementation
|   `-- evaluate_tracking.py # tracking evaluation CLI
|-- src/
|   `-- vn_traffic/          # perception, analytics, evidence, reasoning contracts
|-- tests/                   # dataset, metrics, pipeline, and analytics tests
|-- app.py                   # Streamlit dashboard over a pipeline run directory
|-- detect.py                # backward-compatible root CLI
|-- run_pipeline.py          # repository-local MVP pipeline CLI
|-- environment.yml          # audited Conda environment
|-- pyproject.toml
`-- LICENSE                  # AGPL-3.0-only
```

Large datasets, model weights, generated videos, and runtime outputs are kept
outside version control. The local workspace retains the audited v2 source,
superseded v4 provenance set, current v5 dataset, required baseline/v5
checkpoints, and latest run15-run16 acceptance outputs (with run13-run14 kept
locally as the preceding evidence-schema history). Temporary extraction
sets, superseded smoke/finetune runs, invalid analytics runs, and legacy output
videos are intentionally removed after their lightweight manifests or findings
have been recorded.

## Quick start

### Environment

```powershell
conda env create -f environment.yml
conda activate traffic

python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
python -m unittest discover -s tests -v
```

The recorded environment uses Python 3.10.20, PyTorch 2.6.0 with CUDA 12.4,
Ultralytics 8.4.115, and an RTX 3050 Laptop GPU with 6 GB VRAM.

### Detection

Always pass an explicit model path. The historical CLI default still points to
the legacy v2 checkpoint for backward compatibility.

```powershell
# One image
python detect.py test_image.jpg --model path/to/best.pt --conf 0.5

# Directory of images
python detect.py datasets/vn_images --model path/to/best.pt --conf 0.4

# Video
python detect.py datasets/raw_videos/traffic_jam.mp4 --model path/to/best.pt --conf 0.3
```

Each invocation creates the next `output/runN/` directory and stores annotated
media plus `detections.csv`.

### Offline detection and tracking pipeline

The new MVP path uses one YOLO instance for detection and ByteTrack. It writes
the stable artifact contract documented in [Output schema](docs/output_schema.md).

```powershell
# Validate paths without loading the model
python run_pipeline.py --dry-run

# Run after GPU training is complete
python run_pipeline.py `
  --source datasets/raw_videos/traffic_normal.mp4 `
  --model runs/detect/research/yolov8s_v5_seed0/weights/best.pt

# Short integration check
python run_pipeline.py --max-frames 30 --imgsz 640
```

The default config references the validation-selected v5 checkpoint. Override
`--model` to run another checkpoint without editing the YAML.

The first real-model integration check processed 30 frames of
`traffic_normal.mp4` and produced all four required artifacts: an annotated
video, 658 track rows, an intentionally empty Stage 1 event stream, and a
completed run manifest. It measured 2.35 end-to-end FPS at `imgsz=640`, but
this short cold-start run is a functional check rather than a formal latency
benchmark. Its lightweight provenance is stored in
`experiments/pipeline_v5_integration_20260817T115039/run.json`.

### Dashboard

`app.py` is a Streamlit page that reads an existing `output/pipeline/runN/`
directory and shows the current annotated frame, congestion state, an
occupancy/track-count timeline, recent events, and the finished annotated
video. It does not connect to a live camera: the project has no live camera
source, only offline video files processed by `run_pipeline.py`. "Real-time"
here means the dashboard polls a run's own output files while that run is
still writing them: `runner.py` flushes `tracks.csv`/`analytics.csv`/
`events.jsonl` after every frame, rewrites `run.json`'s progress fields about
once per second, and overwrites `latest_frame.jpg` every frame through a temp
file plus an atomic rename, so the dashboard never reads a half-written JPEG.
`latest_frame.jpg` is the actual live view; `annotated.mp4` is not readable
live because most containers only finalize their index when the writer
closes, so it only becomes playable once the run completes and is shown
purely for after-the-fact review.

```powershell
python -m pip install streamlit==1.61.1   # once; see [project.optional-dependencies].dashboard
python -m streamlit run app.py
```

This opens `http://localhost:8501`. Use the sidebar to pick a run (defaults to
the most recently modified one) and the refresh interval; auto-refresh is only
active while that run's `status` is `running`. To see it update live, start a
pipeline run in one terminal and open the dashboard on the same run in
another:

```powershell
python run_pipeline.py --config configs/pipeline/offline_video_uav_gmc.yaml
```

Verified so far: the app boots headless without exceptions and serves HTTP
200 when reading real run output (`output/pipeline/run27`, `run28`). A human
has not yet watched the auto-refresh update live in a browser against an
in-progress run; treat that specific behavior as implemented but not visually
confirmed until someone does.

### Reproducible training

```powershell
# Validate paths, hashes, Git state, GPU, and the locked test
python scripts/train/train_detector.py `
  --config configs/experiments/yolov8s_v5_seed0.yaml `
  --dry-run

# One-epoch pipeline check; not a research result
python scripts/train/train_detector.py `
  --config configs/experiments/yolov8s_v5_seed0.yaml `
  --smoke

# Full configured run
python scripts/train/train_detector.py `
  --config configs/experiments/yolov8s_v5_seed0.yaml
```

At launch, full runs require a committed, clean worktree. The runner records config,
weights, dataset, manifest and test-lock hashes together with the Git commit,
environment, GPU, and resulting checkpoint hash.

## Evaluation policy

- Select checkpoints and thresholds on validation.
- Use calibration for PTQ calibration and related configuration only.
- Never use locked test for model, tracker, prompt, or backend selection.
- Report precision, recall, mAP50, and mAP50-95 per class and overall.
- Report tracking only after sequence-level evaluator validation.
- Split latency into preprocessing, inference, postprocessing, tracking, VLM,
  and LLM stages.
- Report hardware, precision, warm-up, sample count, and percentile latency.
- Treat smoke runs and legacy leaked runs as diagnostics, never final evidence.

The full measurement contract is defined in
[the benchmark protocol](docs/benchmark_protocol.md).

## Known limitations

- Source videos collected from the web have incomplete provenance and cannot be
  assumed redistributable.
- The dataset remains small and class-imbalanced, especially for trucks, buses,
  and test pedestrians.
- The current locked test covers the available sources; broader geographic,
  weather, night, altitude, and camera-motion coverage is still required.
- The validation-to-test detector gap is severe (`mAP50 0.600` versus `0.148`).
  The locked test contains far smaller objects and different source groups;
  v5 is therefore a functional prototype, not a production-general detector.
- The provenance-controlled tracking result is an integration baseline on
  VisDrone, not a Vietnam-domain or official VisDrone benchmark. Its low recall
  and frequent identity switches limit counting reliability.
- Traffic speed requires camera calibration or a documented approximation.
- The initial congestion thresholds are calibrated on only two demo clips at
  their current resolutions; bbox-union coverage and pixel-speed thresholds may
  not transfer to another camera, crop, or viewpoint. Bbox union is image-plane
  box coverage, not physical road occupancy: boxes include background and the
  pipeline has no segmentation, BEV transform, or camera calibration.
- Line-crossing counts depend on stable ByteTrack identities. Occlusion-driven
  ID switches or fragmentation can cause duplicate or missed counts, and the
  error rate has not yet been measured on the two demo videos.
- VLM/LLM quality, hallucination rate, and quantization effects are not yet
  measured.
- No model has yet been benchmarked on a physical edge NPU in this project.
- Global motion compensation (`analytics.gmc_enabled`) is 2D image-plane
  alignment only (`cv2.findTransformECC`, Euclidean model), not a BEV or
  GPS/IMU-based transform. It can lose lock on hard cuts, very fast pans, or
  low-texture frames; `gmc_consecutive_failures_at_end` in the run summary
  must be checked, not assumed zero.
- `validate_grounding_policy` only checks whether an event's *request*
  references a clip; `run_vlm_case` never actually loads or shows clip frames
  to the VLM. A motion claim on a clip-bearing event is therefore not
  currently caught as ungrounded even though the policy name implies it would
  be.
- The dashboard's live-frame write (`latest_frame.jpg`) can fail on Windows
  due to transient file locks (observed `PermissionError: [WinError 5]` from
  Defender/OneDrive scanning during a real run). This is now handled as
  non-fatal (the frame write is skipped, the run continues), but it means the
  dashboard can occasionally show a stale frame rather than the current one.

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
- [ ] Deferred: export and benchmark detector FP16/INT8 candidates.
- [ ] Deferred: quantize and benchmark the selected VLM and LLM.
- [x] Run a bounded end-to-end UAV benchmark on the RTX host.
- [x] Add a Streamlit dashboard over pipeline run output (headless boot verified; live browser auto-refresh not yet human-confirmed).
- [x] Diagnose the UAV camera-motion ROI failure and implement global motion compensation (`analytics.mode: uav_motion`, `gmc_enabled`), verified on 300 real frames with zero GMC failures and a correct NORMAL→CONGESTED transition.
- [x] Rewrite the dashboard around an atomic live-frame view (`latest_frame.jpg`) after user feedback that the first multi-widget layout was not clean enough.
- [x] Fix the VLM/LLM prompt-copying bug (v1 to v3): eliminated copyable example sentences from both the VLM and LLM prompts, verified with grounded, distinct, analytics-consistent output on two real clips.
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
