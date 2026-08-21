# Project history

Full changelog: completed work, experiments (including rejected ones), and
minor fixes, grouped by theme. The [readme](../readme.md) states only
current results and open priorities; this file is the detailed log behind
it. Full method detail and rationale for any item below is in
[the benchmark protocol](benchmark_protocol.md) unless linked otherwise.

## Dataset

- [x] Audit the Vietnam dataset, identify cross-split leakage, and build
  source-grouped, hash-locked v5 splits -- retained only for field
  validation, see the readme's
  [Field validation](../readme.md#field-validation-vietnam-clips-historical)
  section. VisDrone superseded it as the primary dataset 2026-08-21 (see the
  readme's [Dataset](../readme.md#dataset) section).
- [x] Source and lock VisDrone2019-DET-test-dev (1,610 images, public GT)
  2026-08-21, closing the "repeated selection against val" gap. First read:
  the promoted detector's AP-small gate does not clearly replicate
  (val +0.0223 -> test-dev +0.0027) -- see
  [benchmark protocol](benchmark_protocol.md#visdrone-highres-fine-tune-pilot-and-checkpoint-promotion).
  No equivalent locked split exists yet for tracking/ReID (MOT-val only).

## Detector

- [x] Complete v5 fine-tuning, validation-based checkpoint selection, and
  the one-time locked-test evaluation (historical, Vietnam v5).
- [x] Compare standard, high-resolution, SAHI, and hybrid inference on
  VisDrone-DET; select standard 1280.
- [x] Test an NWD bbox-loss ablation against the small-object gap
  (historical, on Vietnam v5) -- **rejected**, worse than baseline on
  every metric and class.
- [x] Test a P2 detection-head architecture ablation against the same gap
  (historical, on Vietnam v5) -- **rejected**, worse than baseline and
  NWD; required recovering from a training crash (CUDA OOM, a BatchNorm
  corruption at batch=1, and an Ultralytics `resume=True` bug).
- [x] Diagnose a train/infer resolution mismatch on the VisDrone baseline
  checkpoint (trained at 640, infers at 1280) and VRAM-validate a gated
  continuation pilot (batch=2 safe at 3.21/3.74GB; batch=4 rejected at
  ~6.94GB).
- [x] Run the 5-epoch native-1280 continuation and evaluate against the
  frozen gate -- **passed with margin** (AP-small +0.0223, overall AP
  +0.0325), promoted as the UAV pipeline default 2026-08-21.

## Tracking and counting

- [x] Feed the selected detector into ByteTrack once per source frame;
  compare 640 vs. 1280.
- [x] Repair and validate sequence-level class-aware tracking evaluation
  (motmetrics IoU-distance construction fix).
- [x] Integrate TrackEval for HOTA/DetA/AssA; decompose the tracking
  bottleneck as detection-limited, not association-limited.
- [x] Test a BoT-SORT/ReID ablation against ByteTrack -- algorithm switch
  helped (ID switches roughly halved), `model:auto` ReID added
  essentially nothing.
- [x] Fix a bug in `scripts/evaluate_hota.py` that silently dropped
  zero-prediction frames instead of scoring them as false negatives
  (32-33 of 2,846 frames per tracker); moved every HOTA number by <=0.002.
- [x] Confirm the highres-pilot detection gain propagates into tracking
  without touching the tracker (IDF1 +0.023, MOTA +0.089, precision
  +0.064, ID switches -114).
- [x] Test a real pretrained ReID embedding (`yolo26n-reid.onnx`) against
  `model:auto` -- **no improvement**, reconfirms the detection-recall
  bottleneck.
- [x] Derive frame-count and line-crossing ground truth from VisDrone-MOT
  trajectories and measure error.

## Alerts and analytics

- [x] Implement deterministic analytics/event schema with synthetic
  tests; complete ROI/counting-line/congestion acceptance on two demo
  videos.
- [x] Add and synthetic-test a prolonged-stop alert with speed hysteresis
  and gap reset.
- [x] Diagnose the UAV camera-motion ROI failure and implement GMC
  (`analytics.mode: uav_motion`, `gmc_enabled`). Corrected an earlier
  overclaim: the `NORMAL`->`CONGESTED` fix is `uav_motion`'s count-alone
  trigger, not GMC (A/B tested); reconfirmed on the promoted checkpoint
  2026-08-21. Also fixed a failure-count claim that only checked the
  end-of-run streak (`gmc_total_failures` now tracks the run-wide count).
- [x] Build a detection-independent "stalled and dense" signal for
  severe-occlusion jams (`src/vn_traffic/analytics/stillness.py`) --
  automatic `CONGESTED` trigger **rejected and root-caused** (Laplacian
  texture cannot distinguish packed vehicles from any other static
  detailed surface; two further hypotheses also rejected). A **visual
  heatmap** variant (decoupled from the state machine) **works** --
  fixed an EMA-smoothing bug that made it nearly invisible.
- [x] Test two cheap congestion-trigger fixes: lowering detector
  confidence (0.4->0.1, `fixed_camera` mode) is a real, partial, safe
  improvement; dropping the occupancy co-requirement to let count alone
  trigger `CONGESTED` was tried and **rejected** (79% false-positive rate
  on a light-traffic reference clip).
- [ ] Open: the confidence fix alone still leaves the motivating clip
  mostly `NORMAL`. Also open: per-lane/multi-region ROI decomposition and
  verified ego-motion compensation.

## VLM/LLM and evidence

- [x] Freeze VLM/LLM evaluation inputs, JSON/prompt contract v1, add
  two-reviewer annotation templates/validation/adjudication tooling.
- [x] Complete two independent reviewer annotation sets for reasoning
  evaluation v1.
- [x] Fix the VLM/LLM prompt-copying bug (v1 to v3): every run before
  this copied a literal example sentence from the prompt instead of
  describing the actual image; verified grounded on two real clips.
- [x] Add deterministic event keyframe/clip evidence selection; remove
  codec-dependent random seeking; add provenance hashes.
- [x] Add a Streamlit dashboard over pipeline run output (headless boot
  verified; live browser auto-refresh not human-confirmed).
- [x] Run a bounded end-to-end UAV benchmark on the RTX host.
- [ ] Resolve or formally defer the reasoning adjudication queue; does
  not block CV delivery.

## TVLR (paused after Stage B)

- [x] Freeze the offline TVLR feasibility protocol before inspecting its
  oracle result -- excludes detections ByteTrack already recovers,
  reports candidates by score band, protects an internal holdout, forbids
  a real-time claim. See [TVLR protocol](tvlr_protocol.md).
- [x] Run the Stage-B development oracle on 896 VisDrone-MOT frames
  (cached post-NMS proposals, `conf=0.02`, `max_det=3000`, 0 saturated
  frames): 33.8% of ByteTrack-missed tiny/occluded GT recoverable,
  +0.146 recall ceiling, but only 6.1% WAPE improvement under the ideal
  GT-selected oracle and worse on one of three dev sequences -- real
  opportunity, but not an achieved result; false-positive/count-error
  control is the central Stage-C risk. `experiments/tvlr_oracle_dev_v1_20260820/run.json`.
- [ ] **Paused**: Stage C (implementation against frozen baselines,
  precision/HOTA/counting gates, no GT identity) is deprioritized while
  detector/tracking work on VisDrone continues instead.

## Deferred beyond current goal

- [ ] Evaluate a newer Ultralytics architecture generation (e.g. YOLO26)
  as a new baseline; not assumed better or worse than YOLOv8 until
  measured.
- [ ] Export and benchmark detector FP16/INT8 candidates.
- [ ] Quantize and benchmark the selected VLM and LLM.
- [ ] Validate an appropriate physical edge/NPU target.
