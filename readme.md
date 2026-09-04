# Vietnam Traffic Intelligence

Detection, tracking, counting, alerting, and multimodal (VLM/LLM) reporting
for Vietnamese UAV traffic video, running on an NVIDIA RTX 3050 Laptop GPU
(6 GB VRAM).

See [docs/quickstart.md](docs/quickstart.md) for setup and CLI usage,
[docs/output_schema.md](docs/output_schema.md) for the exact artifact
contract, and [docs/reasoning_protocol.md](docs/reasoning_protocol.md) for
the VLM/LLM contract. The active development plan (pipeline completion +
SUMO simulation) is [reports/ke-hoach-pipeline-va-mo-phong.md](reports/ke-hoach-pipeline-va-mo-phong.md).

## The 3 tasks

1. **Detect and count vehicles** from UAV video.
2. **Improve counting reliability with tracking** -- keep vehicle identities
   stable across frames.
3. **Describe traffic conditions and review possible incidents in Vietnamese**
   using pretrained VLM/LLM models. Numeric facts come only from deterministic
   analytics. Incident coverage is hybrid: analytics events trigger review,
   while periodic `visual_scan` clips bypass the detector's five-class gate.
   The VLM supplies a fallible visual assessment; application code, not the
   LLM, maps that assessment to `none`/`review`/`alert`.

## System architecture

```text
camera / UAV / video
        |
        +--> YOLOv8 (5 classes) --> BoT-SORT/ReID --> analytics events --+
        |                                                                |
        +--> periodic visual_scan (class-independent) -------------------+--> raw evidence --> VLM assessment
                                                                                                  |
                                              deterministic event facts --------------------------+
                                                                                                  v
                                                                                       LLM wording only
                                                                                                  |
                                                                                       policy-owned action
```

The LLM and VLM sit outside the per-frame critical path. Event scope is
invoked by selected analytics events or periodic `visual_scan` triggers.

## Current model and pipeline default

- **Detector**: YOLOv8s, fine-tuned in two stages: COCO -> VisDrone2019-DET
  (`runs/detect/baseline/yolov8s_visdrone/weights/best.pt`, mAP50=0.389) ->
  Vietnam v6 (`runs/detect/research/yolov8s_v6_seed0-2/weights/best.pt`, the
  checkpoint the pipeline actually loads; mAP50=0.612, mAP50-95=0.352). v6
  trains from the same VisDrone base in a single pass over the union of the
  Vietnam v5 data and nadir (90-degree) UIT-ADrone samples, rather than
  fine-tuning on top of v5, so it gains the overhead viewpoint without
  forgetting the oblique one. Validation and test splits are unchanged from
  v5, so the numbers are directly comparable: v5 scored mAP50=0.598,
  mAP50-95=0.340 on the same split. The VisDrone-stage checkpoint is kept as
  the base weights the experiment configs fine-tune from, not as a separate
  pipeline option.
- **Tracker**: BoT-SORT with ReID (`botsort_reid_lowprox.yaml`), chosen by
  direct visual review of ID retention across occlusion on real UAV
  footage -- see `configs/pipeline/offline_video.yaml` for the full
  reasoning in its comments.
- **VLM/LLM**: Qwen3-VL-2B-Instruct produces a cited visual assessment and
  Qwen3-0.6B writes Vietnamese summary/recommendation text. Numeric facts and
  `action.level` are assembled and validated by application code. The incident
  ontology is event-based (`collision`, `fire_or_smoke`, `road_obstruction`,
  etc.), not limited to detector class names.
- **Private media**: datasets, source videos, and video-specific configs/manifests
  stay local and are ignored by Git. Public pipeline configs use `<video_path>`;
  provide the source explicitly with `--source`.
- **Full pipeline (`reasoning.enabled: true`)**: a pipeline config can wire
  the VLM directly into the same `run_pipeline.py` invocation, no separate
  CLI round trip. `traffic_window` (recommended for routine summaries)
  describes fixed time windows with one VLM call per window and writes
  `<run_dir>/traffic_windows.jsonl`. `event` describes each
  `visual_scan`/`congestion_transition`/`prolonged_stop` trigger with VLM +
  LLM and an evidence-reference audit trail in `<run_dir>/reasoning.jsonl`.
  The public example is `configs/pipeline/offline_video_reasoning.yaml`.

## Incident review: measured limits and hybrid coverage

Incident judgement was measured against UIT-ADrone frame-level anomaly masks.
Neither tested path reached usable quality, so periodic visual review is an
experimental candidate generator, not a claim of reliable accident detection.
The hybrid design fixes a coverage bug—no detector event no longer means no VLM
input—but does not improve the measured VLM classifier accuracy by itself.

**Path 1 -- let the VLM judge.** The VLM was shown real multi-frame clips
(not a single keyframe, which carries no motion information at all) and
asked to decide. On a balanced 66-clip probe it reached recall 45.5%,
precision 53.6%, F1 49.2%. A naive "always alert" rule scores 100% recall /
50% precision on the same balanced set, so the model's decisions carried
little information beyond chance. Its *descriptions* of the same clips were
specific and grounded -- the failure is in the judgement, not the seeing.

**Path 2 -- let deterministic analytics judge.** `prolonged_stop` was
repaired (see below) and swept against the ground-truth masks of three
UIT-ADrone videos. It fired 14 times across 1745 frames, and its precision
sat at or below the label base rate on every video (0.286 vs a 0.354 base
rate; 0.600 vs 0.598; 0.500 from only two firings vs 0.176), i.e. no
information. Per-segment coverage was 3 of 11 labelled segments.

**Why path 2 fails is upstream of the analytics.** Tracing DJI_0084's
labelled segment back to source frames shows the anomaly is a garbage truck
parked at the top frame edge while a worker loads it. The detector emits no
truck-sized detection there for the whole segment (frames 0-199; the first
is at frame 155, confidence 0.11). The object is cut off by the frame
boundary, dark, and its silhouette is broken by piled waste. No analytics
threshold can surface an anomaly the detector never reports -- and this is
the general shape of the problem, since an anomalous object is by definition
an unusual-looking one.

Note the two paths fail for *different* reasons: the VLM reads raw clip
pixels and bypasses the detector entirely, so detector recall does not
explain its result.

**Not disproven:** LoRA fine-tuning of the VLM. The one attempt was
invalidated by a training bug (examples were fed in contiguous class blocks
without shuffling, so the run ended on a long block of negatives and the
adapter collapsed to predicting "not_observed" for all 32 evaluated
positives). The bug is fixed but the experiment was not re-run. A larger VLM
and more frames per clip were also never tested -- both were ruled out by the
6 GB VRAM budget, not by measurement.

Tooling for reproducing all of the above lives in `scripts/analytics/`
(`diagnose_prolonged_stop.py`, `calibrate_prolonged_stop.py`).

## Known limitations

- **Tracking under occlusion**: ID loss at a gantry/overhead structure on
  real UAV footage measured at ~7-8% retention. ReID and motion-model
  (`track_buffer`/`match_thresh`) tuning were both tried; ReID was kept
  based on direct visual review of the actual output video despite an
  automated candidate-switch proxy metric disagreeing.
- **Detection silence vs. a genuinely clear road**: if the detector
  produces zero detections for a sustained period, `perception_status`
  flips to `detection_silence` and any congestion-state transition to
  `NORMAL` during that window is reported as `UNKNOWN` instead, so a silent
  detector failure cannot be misread downstream as "no traffic."
- Traffic speed and counting are image-plane measurements: no camera
  calibration or BEV transform is applied, so results are not physically
  calibrated flow.
- VLM/LLM *report* quality and hallucination rate are not formally measured
  (no frozen human-annotated evaluation set). Its *anomaly-judgement*
  quality has been measured and is near chance -- see "Incident review: measured limits and hybrid coverage" above.
- **Detector recall on unusual-looking objects** remains a binding constraint
  for counting, tracking, and analytics-triggered alerts. Periodic
  `visual_scan` bypasses that gate for VLM review, but the measured 2B VLM
  incident judgement is weak and must not be treated as ground truth.
- `prolonged_stop` fires rarely even after repair, because a stopped vehicle
  must also be tracked continuously for `min_duration_s`; observed track
  lifetimes (median ~2.3 s after the detector-confidence fix) leave little
  headroom above a 5 s requirement.
- Global motion compensation (`analytics.mode: uav_motion`,
  `gmc_enabled: true`) is 2D image-plane alignment only, not BEV or
  GPS/IMU-based, and can lose lock under a hard cut, fast pan, or
  low-texture frames.
- The dashboard's live-frame write can fail on Windows due to transient
  file locks (handled as non-fatal).
- No model has been benchmarked on a physical edge NPU; quantization and
  physical edge/NPU deployment are out of current scope.

## License

Code in this repository is licensed under
[GNU AGPL-3.0-only](LICENSE). Dataset, video, pretrained-weight, and third-party
asset licenses remain separate and must be verified before redistribution or
commercial use.

## Acknowledgements

This project uses Ultralytics YOLO, PyTorch, OpenCV, pandas, BoT-SORT/
ByteTrack concepts, and Qwen (Qwen3-VL, Qwen3) from Alibaba Cloud.
