# Vietnam Traffic Intelligence

Leakage-controlled traffic detection, tracking, analytics, and multimodal
deployment-readiness research for fixed cameras and UAV video.

The current system fine-tunes YOLOv8 on Vietnamese traffic scenes and provides
the foundation for ByteTrack-based tracking, structured traffic analytics, and
event-driven VLM/LLM interpretation. Development and measurement currently run
on an NVIDIA RTX 3050 Laptop GPU with 6 GB VRAM. Edge and NPU support is a
roadmap and readiness target, not a claim of deployment on physical edge
hardware.

<!-- IMAGE PLACEHOLDER: hero image showing detector, tracks, traffic analytics,
VLM/LLM report, and the optional future edge/NPU path. -->

---

## Why this project

Vietnamese road scenes are dense, dominated by small motorcycles and
pedestrians, and frequently affected by occlusion and camera motion. A useful
system must do more than draw boxes: it must preserve identities, aggregate
traffic state, explain noteworthy events, and report the accuracy and runtime
cost of every optimization honestly.

This repository therefore separates four concerns:

1. dataset integrity and leakage-controlled evaluation;
2. object detection and multi-object tracking;
3. traffic analytics and multimodal interpretation;
4. deployment readiness through export, quantization, and hardware-specific
   benchmarking.

## What it does

| Capability | Current implementation | Status |
|---|---|---|
| Image and video detection | YOLOv8, five Vietnamese traffic classes | Implemented |
| Multi-object tracking | Ultralytics ByteTrack integration and custom tracker configuration | Implemented; evaluator repaired, v5 benchmark pending |
| Traffic density routing | Preprocessing module under `src/preprocessing` | Prototype |
| Structured traffic analytics | ROI occupancy, trajectories, directional counts, and congestion events | Implemented; initial two-video acceptance passed |
| VLM scene understanding | Representative frames or event clips, not every frame | Architecture defined |
| LLM reasoning and reports | Event-driven summaries using structured analytics plus VLM evidence | Architecture defined |
| Detector quantization | FP16/INT8 export and accuracy-latency evaluation | Planned |
| VLM/LLM quantization | Backend-specific FP16/INT8 or weight-only evaluation | Planned |
| Physical edge/NPU execution | Target-specific validation | Not yet measured |

See [the multimodel architecture](docs/multimodel_architecture.md) for component
boundaries and the intended event-driven VLM/LLM path.

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

<!-- IMAGE PLACEHOLDER: complete 16:9 architecture block diagram. Use solid
lines for the current host path and dashed lines for future edge/NPU paths. -->

## Research status

| Area | Status | Evidence |
|---|---|---|
| Audited Python/CUDA environment | Complete | [Environment](docs/environment.md) |
| Legacy Vietnam v2 leakage audit | Complete; invalid for scientific claims | [Dataset protocol](docs/dataset_protocol.md) |
| Source-grouped Vietnam v5 dataset | Complete | `configs/datasets/vietnam_v5.yaml` |
| Content-addressed locked test | Complete, 176 images | `manifests/datasets/vietnam_v5/test_lock.json` |
| YOLOv8s v5 smoke run | Complete | `experiments/yolov8s_v5_seed0_smoke_20260817T100534/run.json` |
| YOLOv8s v5 full fine-tuning | Complete, 30 epochs | `experiments/yolov8s_v5_seed0_20260817T100644/run.json` |
| Offline-video CLI and artifact contract | Integrated with v5 checkpoint | [Output schema](docs/output_schema.md) |
| Deterministic analytics state machine | 32 tests passed; bbox-union two-video acceptance passed | [Output schema](docs/output_schema.md) |
| Tracking evaluator | IoU association repaired and unit-tested; v5 benchmark pending | [Benchmark protocol](docs/benchmark_protocol.md) |
| Export and quantization benchmark | Not started | [Benchmark protocol](docs/benchmark_protocol.md) |
| VLM/LLM implementation | Not started | [Multimodel architecture](docs/multimodel_architecture.md) |

No locked-test metric is reported while model, threshold, tracker, prompt, or
quantization decisions are still being made.

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

### VisDrone initialization baseline

YOLOv8s was initialized from COCO and fine-tuned on VisDrone2019-DET. The best
validation epoch in the stored training log is epoch 74.

| Model | Validation scope | Best epoch | mAP50 | mAP50-95 | Research use |
|---|---|---:|---:|---:|---|
| YOLOv8s VisDrone baseline | VisDrone2019-DET validation, 10 classes | 74 | 0.389 | 0.225 | Initialization checkpoint |

### Historical Vietnam v2 result

| Model | Input | Batch | Freeze | Optimizer | mAP50 | mAP50-95 | Validity |
|---|---:|---:|---:|---|---:|---:|---|
| YOLOv8s Vietnam v2 | 1280 | 8 | 10 | `auto` | 0.745 | 0.481 | **Invalid for scientific comparison: split leakage** |

The table records what was actually stored in `args.yaml`; it corrects the old
README claim that this run used `freeze=0`.

### Leakage-controlled Vietnam v5

The primary run uses full fine-tuning (`freeze=0`) from the VisDrone checkpoint.

| Parameter | Value |
|---|---:|
| Epochs | 30 |
| Input size | 1280 |
| Batch | 4 |
| Optimizer | AdamW |
| Initial learning rate | 0.0005 |
| Weight decay | 0.0005 |
| Mosaic / MixUp | 1.0 / 0.3 |
| Seed | 0 |
| Deterministic mode | Enabled |
| AMP | Enabled |
| Model selection split | Validation only |

The one-epoch smoke run completed successfully at 640 pixels and batch 2. It
verified the data loader, class remapping, explicit AdamW configuration, CUDA
execution, and locked-test guard. Its validation values (`mAP50=0.120`,
`mAP50-95=0.040`) are pipeline diagnostics, not research results.

Final v5 validation and locked-test results will be added only after training,
model selection, and calibration are complete.

The full 30-epoch run completed successfully. Epoch 29 was selected by the
highest validation `mAP50-95`; no locked-test samples were used.

| Selected epoch | Precision | Recall | mAP50 | mAP50-95 | Checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|
| 29 | 0.762 | 0.504 | 0.600 | 0.344 | `729c66e676345e9c...` |

These values are validation results. Locked-test metrics remain pending until
the complete detector, tracker, calibration, and runtime configuration is
frozen.

<!-- IMAGE PLACEHOLDER: v5 training curves generated by Ultralytics. -->

<!-- IMAGE PLACEHOLDER: per-class PR curves and confusion matrix. -->

<!-- IMAGE PLACEHOLDER: qualitative Vietnamese traffic detections covering
small motorcycles, dense traffic, pedestrians, and failure cases. -->

## Tracking and analytics

ByteTrack is available through `src/detection/tracker.py` and
`bytetrack_custom.yaml`. Historical files named `tracking_metrics_*.csv` must
not be used as reported results: the original evaluator inverted an already
distance-form IoU matrix and therefore produced invalid associations.

The repaired evaluator now uses `1-IoU` directly, applies the configured IoU
gate, includes prediction-only frames, and produces a combined OVERALL
accumulator instead of averaging sequence metrics. Five synthetic tests cover
perfect overlap, gated non-overlap, threshold behavior, perfect tracking, and
prediction-only false positives.

An end-to-end diagnostic over seven existing VisDrone prediction files
completed after the repair:

| Scope | MOTA | MOTP distance | IDF1 | ID switches | Validity |
|---|---:|---:|---:|---:|---|
| 7 VisDrone-MOT sequences, OVERALL | 0.123 | 0.272 | 0.339 | 203 | Legacy integration check only |

The prediction files used by this check lack complete model/config provenance,
so these values are not a v5 tracking result and must not be used for model
comparison.

Stage 2 adds a deterministic analytics engine under `src/vn_traffic/analytics`.
It maintains per-track trajectories, counts one crossing per direction and
track ID, measures unique bbox-union coverage inside the ROI, estimates centroid
speed in pixels per second, and applies temporal hysteresis to `NORMAL`,
`DENSE`, and `CONGESTED`. Geometry is normalized in YAML and kept separate from
the state machine.

Initial acceptance used the v5 checkpoint at `imgsz=640`, confidence 0.4, and
the initial center-corridor ROI/counting line in
`configs/pipeline/offline_video.yaml`. The jam clip was processed in full; the
normal clip used its first 300 frames. Overlay frames were visually inspected
to confirm that the normalized ROI and line intersect the intended road
corridor. These are calibration/demo videos, not the locked test.

| Acceptance clip | Evaluated span | Bbox-union occupancy median (range) | Speed median px/s | State timeline | Result |
|---|---:|---:|---:|---|---|
| `traffic_jam.mp4` | 283 frames / 11.3 s | 0.589 (0.488-0.724) | 78.0 | `NORMAL` to `CONGESTED` at 2.12 s; 230 congested frames | Pass |
| `traffic_normal.mp4` | 300 frames / 10.0 s | 0.135 (0.069-0.219) | 104.6 | `NORMAL` for 300/300 frames | Pass |

The thresholds were selected only after inspecting these two timelines. This
demonstrates deterministic separation for the current scenes and resolution;
it is not evidence that the thresholds generalize to other cameras. Crossing
events were generated in both runs, but count accuracy is not reported because
the clips have no counting ground truth and tracker ID error is not yet known.

The original Stage 2 implementation summed bbox areas and double-counted
overlap. Local run2-run8 analytics artifacts are retained with an
`INVALID_ANALYTICS.json` sidecar and must not be cited; their `tracks.csv`
inputs remain reusable. Run1 predates analytics and is unaffected. Corrected
metric diagnostics start at run9; final acceptance evidence is run11-run12
with analytics schema version 2. The new schema deliberately renames the metric
to `bbox_union_occupancy` instead of silently changing legacy `occupancy`
semantics.

Raster-grid selection was measured over all 583 acceptance frames. Grid 1 is
the default because its 3.63-5.51 ms/frame cost is small relative to the current
detector/tracker pipeline. Grid 2 reduced the metric cost to 0.89-1.22 ms/frame
with maximum absolute error 0.007 versus grid 1; grid 4 cost 0.26-0.29 ms/frame
with maximum error 0.021. Both candidates preserved the two demo timelines and
remain configurable options for future edge benchmarks.

| Metric | Current status |
|---|---|
| HOTA | Pending TrackEval integration |
| IDF1 | Evaluator repaired; provenance-controlled v5 run pending |
| MOTA | Evaluator repaired; provenance-controlled v5 run pending |
| ID switches | Evaluator repaired; provenance-controlled v5 run pending |
| Detector + tracker latency | Pending benchmark |

<!-- IMAGE PLACEHOLDER: track trajectories, line crossings, and traffic-density
overlay on a representative video frame. -->

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

<!-- IMAGE PLACEHOLDER: accuracy-versus-latency and model-size comparison for
FP32, FP16, and INT8 after export benchmarks exist. -->

## Repository layout

```text
.
|-- configs/
|   |-- datasets/            # audited dataset metadata
|   `-- experiments/         # reproducible experiment definitions
|-- docs/                    # protocols, architecture, and environment evidence
|-- experiments/             # lightweight run manifests and hashes
|-- manifests/datasets/      # leakage audits and locked-test manifests
|-- scripts/
|   |-- data/                # audit, materialization, overlap, and lock tools
|   |-- train/               # provenance-aware detector training
|   |-- detect.py            # image, directory, and video inference
|   `-- ...                  # legacy experiment/evaluation scripts
|-- src/
|   |-- detection/           # detector and tracker modules
|   |-- preprocessing/       # density-routing prototype
|   |-- vn_traffic/          # new single-model offline pipeline package
|   `-- pipeline.py          # current pipeline entry point
|-- tests/                   # unit tests for dataset and training safeguards
|-- detect.py                # backward-compatible root CLI
|-- run_pipeline.py          # repository-local MVP pipeline CLI
|-- environment.yml          # audited Conda environment
|-- pyproject.toml
`-- LICENSE                  # AGPL-3.0-only
```

Large datasets, model weights, generated videos, and runtime outputs are kept
outside version control.

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
- Tracking metrics currently stored at repository root are invalid and retained
  only as historical artifacts.
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

## Roadmap

- [x] Audit the legacy dataset and identify cross-split leakage.
- [x] Build source-grouped train/calibration/validation/test splits.
- [x] Lock test content by image and label hashes.
- [x] Add reproducible detector training and smoke validation.
- [x] Complete v5 fine-tuning and validation-based checkpoint selection.
- [ ] Freeze the complete pipeline and run locked-test evaluation.
- [ ] Repair and validate sequence-level tracking evaluation.
- [x] Implement deterministic analytics and event schema with synthetic tests.
- [x] Complete initial ROI, counting-line, and congestion acceptance on two demo videos.
- [ ] Add event-driven VLM and LLM modules.
- [ ] Export and benchmark detector FP16/INT8 candidates.
- [ ] Quantize and benchmark the selected VLM and LLM.
- [ ] Run end-to-end benchmarks on the RTX host.
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
