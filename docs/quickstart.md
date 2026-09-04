# Quickstart

Operational reference: environment setup, CLI usage, the dashboard, and the
repository layout. For research objectives, methodology, and results, see
[the readme](../readme.md).

## Environment

```powershell
conda env create -f environment.yml
conda activate traffic

python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
python -m unittest discover -s tests -v
```

The recorded environment uses Python 3.10.20, PyTorch 2.6.0 with CUDA 12.4,
Ultralytics 8.4.115, and an RTX 3050 Laptop GPU with 6 GB VRAM. Full audited
package/driver versions: [environment](environment.md).

## Three ways to run this

- **Detection only** -- `detect.py` (below): one YOLO pass over an
  image/directory/video, no tracking, no analytics, no VLM/LLM.
- **Pipeline, no VLM/LLM** -- `run_pipeline.py` with the default config
  (`reasoning.enabled: false`, the default): detection + BoT-SORT tracking +
  analytics (counting, congestion state, `prolonged_stop`) + evidence
  export. The default policy also creates detector-independent `visual_scan`
  clips every five seconds; it still does not load a VLM unless reasoning is
  enabled. This is what every config under `configs/pipeline/` ran before
  `reasoning:` existed, and still what most of them run.
- **Full pipeline (+ VLM; event scope also uses LLM)** -- use the public
  `configs/pipeline/offline_video_reasoning.yaml` config and pass local media
  through `--source <video_path>`. Its `event` scope reviews periodic
  `visual_scan` plus analytics events with VLM then LLM and writes
  `reasoning.jsonl`. For routine summaries, set `scope: traffic_window`; that
  path uses one VLM call per fixed window and writes `traffic_windows.jsonl`.

## Detection (single image/video CLI)

`--model` defaults to `runs/detect/research/yolov8s_v6_seed0-2/weights/best.pt`
(the validation-selected v6 checkpoint, see `scripts/detect.py`) -- pass
`--model` explicitly to run a different checkpoint.

```powershell
# One image -- uses the default checkpoint above
python detect.py <image_path> --conf 0.5

# Directory of images
python detect.py <image_directory> --conf 0.4

# Video
python detect.py <video_path> --conf 0.3

# To use a different checkpoint, pass --model explicitly:
python detect.py <image_path> --model runs/detect/baseline/yolov8s_visdrone/weights/best.pt
```

Each invocation creates the next `output/runN/` directory and stores annotated
media plus `detections.csv`.

## Offline detection and tracking pipeline

The MVP path uses one YOLO instance for detection and tracking (BoT-SORT by
default -- see `configs/pipeline/offline_video.yaml`). It writes the
stable artifact contract documented in [Output schema](output_schema.md):
`tracks.csv`, `analytics.csv`, `events.jsonl`, `evidence.jsonl` + `evidence/`,
`summary.json`, `annotated.mp4`, `latest_frame.jpg`, and `run.json`.

```powershell
# Validate local paths without loading the model.
python run_pipeline.py `
  --source <video_path> `
  --model <model_path> `
  --dry-run

# Detection + tracking + analytics. VLM/LLM stays off in the default config.
python run_pipeline.py `
  --source <video_path> `
  --model <model_path>

# Full pipeline (+ VLM + LLM), using a public config with no pinned video.
python run_pipeline.py `
  --config configs/pipeline/offline_video_reasoning.yaml `
  --source <video_path> `
  --model <model_path>

# Short integration check
python run_pipeline.py `
  --source <video_path> `
  --model <model_path> `
  --max-frames 30 `
  --imgsz 640
```

The public configs intentionally contain `<video_path>` rather than a private
media location. Always provide `--source`; provide `--model` too when your
checkpoint is stored elsewhere. Video-specific configs and manifests remain
local and are excluded by `.gitignore`.

`analytics.mode: uav_motion` + `gmc_enabled: true` remains available for a
moving/panning camera. Fixed-camera analytics is the default.

`stillness_heatmap.enabled: true` (top-level, independent of `analytics.*`)
tints `annotated.mp4` wherever a region is both visually dense and
near-motionless -- a real, validated visual aid for severe-occlusion jams
the detector cannot resolve into boxes. It is a human-facing visualization
only, not an automatic alert: a fixed-threshold automatic trigger for this
same failure mode was tried in `analytics.stillness_enabled` and rejected
(measured no clear improvement on a static-camera test scene).

## Dashboard

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
another.

Verified so far: the app boots headless without exceptions and serves HTTP
200 when reading real run output. A human has not yet watched the
auto-refresh update live in a browser against an in-progress run; treat that
specific behavior as implemented but not visually confirmed until someone
does.

## Reproducible training

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

At launch, full runs require a committed, clean worktree. The runner records
config, weights, dataset, manifest and test-lock hashes together with the Git
commit, environment, GPU, and resulting checkpoint hash.

## VLM/LLM reasoning

### Full pipeline: VLM wired in

Add a `reasoning:` block to a pipeline config to have `run_pipeline.py`
describe traffic conditions itself, no separate CLI round trip. Two
`scope` values pick genuinely different code paths (see
`src/vn_traffic/reasoning/traffic_window.py`'s module docstring for the
reasoning behind the split):

- **`traffic_window` (recommended for routine summaries)** -- describes the whole run in
  fixed `window_seconds` windows, regardless of whether any event fired.
  One VLM call per window, no LLM tier, no incident framing -- just
  `traffic_state`/`observations`/`confidence`/`limitations` grounded in
  analytics numbers already computed for free (vehicle counts, occupancy,
  stopped-track count) plus one representative frame per window.
- **`event`** -- describes each qualifying `visual_scan`/
  `congestion_transition`/`prolonged_stop` event individually (VLM then LLM, evidence-refs audit
  trail). Appropriate for an incident record with provenance, not for a
  routine traffic-condition summary -- see its own subsection below.

```yaml
# Example reasoning block for a local pipeline config
reasoning:
  enabled: true
  scope: traffic_window
  window_seconds: 15.0
  max_long_edge: 768
  vlm_model_dir: models/qwen3-vl-2b-instruct
  vlm:
    model_id: Qwen/Qwen3-VL-2B-Instruct
    revision: 89644892e4d85e24eaac8bacfd4f463576704203
    dtype: float16
    do_sample: false      # retries past the first attempt still force
    max_new_tokens: 480    # sampling -- see run_window_vlm's docstring
    max_attempts: 2
```

The prompt asks for 3 observations per window (density/speed, vehicle-
level visual detail, context/behavior -- lane markings, signage, lighting,
pedestrians/motorcycles if present); `max_new_tokens: 480` and
`max_long_edge: 768` were raised from an initial 192/640 after the first
pass read as too sparse -- see `_window_prompt_text()`'s docstring in
`traffic_window.py` for the exact wording.

```powershell
python run_pipeline.py --config <pipeline_config.yaml> --source <video_path> --model <model_path>
```

This writes `<run_dir>/traffic_windows.jsonl`, one record per window, plus
`<run_dir>/traffic_windows_report.txt` for human reading:
window stats (`vehicle_counts`, `occupancy`, `stopped_tracks`,
`motion_state`, `representative_keyframe`) plus `vlm` (`contract_status`,
`assessment`). Windowing/aggregation reads only files perception+analytics
already wrote (`analytics.csv`, `events.jsonl`) -- no extra model call, no
re-decode beyond grabbing one representative frame per window directly
from the source video.

**Measured on this machine** at the current settings (`max_new_tokens: 480`,
3 observations/window, run113, a fresh 60s/1429-frame video, RTX 3050
Laptop 6 GB): 3 windows, VLM load 23.1s + generate 29.2s + 24.1s + 74.1s
(this last one needed a retry) -- **~150s total for the reasoning stage**.
At the earlier, leaner settings (`max_new_tokens: 192`, 2 observations,
an earlier local 667-frame/22.2s validation clip): 2 windows, both valid on the first
attempt, load 14.3s + generate 8.4s + 6.1s -- ~29s total. Either way this
is still one to a few minutes per video, not the old per-event path's tens
of minutes. Compare to `scope: event` on the same video: 8 `prolonged_stop` cases (one
per stopped vehicle in the same jam, largely redundant), each up to 3
retries at ~260s/attempt observed once VRAM-resident and not reloading per
case -- tens of minutes. The dominant cost in the old per-event path was
genuine RTX 3050 Laptop generation throughput for 384 new tokens, not a
reload/offload bug (confirmed: `load_vlm` uses `device_map="cuda"`, a hard
single-GPU placement that would OOM rather than silently offload, and the
model is loaded once and reused across cases either way); `window_seconds`
batching removes the redundant calls instead, which is what actually
closed the gap.

VLM/system-RAM load time varies a lot run to run depending on whether the
~4.3 GB checkpoint is already in the Windows file cache: measured anywhere
from 8s to 120s across repeated runs this session with no code change,
tracking free system RAM at load time -- see the paging-file note below.

### Full pipeline: `scope: event` (incident record with provenance)

Use this instead of `traffic_window` when the goal is a per-event record
with an evidence-refs audit trail (e.g. feeding a downstream alert/report
system that needs to cite exactly which frames/clip support a claim), not
a routine traffic-condition summary:

```yaml
# visual_scan is a review request, not an incident claim. Include it in both
# lists so event scope receives a temporal clip even when analytics is silent.
evidence:
  enabled: true
  visual_scan_enabled: true
  visual_scan_interval_s: 5.0
  keyframe_event_types: [visual_scan, line_crossing, congestion_transition, prolonged_stop]
  clip_event_types: [visual_scan, congestion_transition, prolonged_stop]
  pre_event_s: 2
  post_event_s: 3

reasoning:
  enabled: true
  scope: event   # default when scope is omitted
  # event_types defaults to [visual_scan, congestion_transition, prolonged_stop] --
  # line_crossing is a counting event (dozens-hundreds per run) that a
  # visual description does not add value to; override to include it if a
  # specific investigation needs it.
  prompts: prompts_v9.yaml
  vlm_model_dir: models/qwen3-vl-2b-instruct
  llm_model_dir: models/qwen3-0.6b
  vlm:
    model_id: Qwen/Qwen3-VL-2B-Instruct
    revision: 89644892e4d85e24eaac8bacfd4f463576704203
    dtype: float16
    clip_sample_frames: 3
    max_new_tokens: 384
  llm:
    model_id: Qwen/Qwen3-0.6B
    revision: c1899de289a04d12100db370d81485cdf75e47ca
    dtype: float16
    thinking: false
    max_new_tokens: 512
  generation:
    do_sample: true
    seed: 0
    max_attempts: 3
```

Writes `<run_dir>/reasoning.jsonl`, one record per qualifying event:
`{case_id, event_id, event_type, vlm, llm}`, where `vlm`/`llm` are the same
`contract_status`/`assessment`(`report`) shape `run_vlm.py`/`run_llm.py`
write standalone (see `src/vn_traffic/reasoning/pipeline_stage.py`). The
VLM loads once and is reused across all qualifying events in the run, then
is freed before the LLM loads. The LLM generates wording only;
`action.level` is derived deterministically (`alert` for an observed collision,
overturn, or fire/smoke; `review` for other observed/uncertain findings;
`none` for `not_observed`). Periodic scanning can be expensive on the 6 GB
target GPU, so tune the interval against required coverage and throughput.

**Measured cost**: on a local validation clip (8 `prolonged_stop` cases -- one per
stopped vehicle in a single jam, largely describing the same scene), each
case took up to ~260s to generate 384 tokens once VRAM-resident -- tens of
minutes for the whole run. Deliberate: this path optimizes for audit
completeness (every event gets its own cited evidence), not for wall-clock
cost, which is exactly why `traffic_window` is recommended for
"just describe the traffic" instead.

### Frozen development cases

`scripts/reasoning/run_vlm.py` and `scripts/reasoning/run_llm.py` are
standalone CLIs over one frozen development case (event + evidence record
from a completed pipeline run, frozen via
`scripts/reasoning/freeze_evidence_set.py`). Use these instead of
`reasoning.enabled` when a case needs to stay byte-identical across runs for
calibration or audit (e.g. prompt-version comparisons) -- the manifest locks
the exact evidence file hashes, so the same case can be replayed after the
underlying pipeline run's outputs might otherwise have changed. Both need a
local model directory (`models/qwen3-vl-2b-instruct`, `models/qwen3-0.6b`)
-- weights are not downloaded automatically. `--dry-run` validates the case
(evidence hashes, model IDs) without loading either model.

```powershell
# Validate a case without loading the VLM
python scripts/reasoning/run_vlm.py `
  --config configs/reasoning/development_v1.yaml `
  --case-id development-0001 `
  --dry-run

# Run the VLM, then feed its validated result into the LLM
python scripts/reasoning/run_vlm.py `
  --config configs/reasoning/development_v1.yaml `
  --case-id development-0001 `
  --model-dir models/qwen3-vl-2b-instruct `
  --output output/reasoning/development-0001-vlm.json

python scripts/reasoning/run_llm.py `
  --config configs/reasoning/development_v1.yaml `
  --case-id development-0001 `
  --vlm-result output/reasoning/development-0001-vlm.json `
  --model-dir models/qwen3-0.6b `
  --output output/reasoning/development-0001-llm.json
```

Both models must load sequentially on a 6 GB GPU (`execution_policy:
sequential_load_run_unload`); a single-image VLM call alone measured a peak
of ~5.7 GB VRAM (see
`experiments/qwen3_vl_2b_dev_smoke_20260817/run.json`), and Windows can also
fail to load either model with "paging file is too small" if free system
RAM is low at the time (unrelated to VRAM -- close memory-heavy
applications and retry). Measured on 2026-09-03: with free system RAM down
to ~880 MB (other running applications, not this project), the failure mode
was a raw segfault (exit 139) inside `AutoModelForMultimodalLM.from_pretrained`
rather than a clean Python "paging file" error -- same root cause, different
symptom; check free RAM first if a VLM/LLM load dies with no Python
traceback at all. See [the reasoning protocol](reasoning_protocol.md) for
the two-stage contract and prompt versions.

## Repository layout

```text
.
|-- configs/
|   |-- datasets/            # audited dataset metadata
|   |-- experiments/         # detector training config (current model)
|   |-- pipeline/            # offline-video runtime configuration
|   `-- reasoning/           # versioned VLM/LLM prompts
|-- docs/                    # quickstart, output schema, reasoning protocol
|-- experiments/             # current model + VLM/LLM smoke-test run manifests
|-- manifests/
|   |-- datasets/            # detector test-split lock
|   |-- measurement/         # per-video ROI/counting-line manifests
|   `-- reasoning/           # content-addressed VLM/LLM input locks
|-- scripts/
|   |-- data/                # test-split lock tool (train_detector.py preflight)
|   |-- reasoning/           # evidence-set freezing, VLM/LLM CLIs
|   |-- train/               # provenance-aware detector training
|   `-- detect.py            # image, directory, and video inference
|-- src/
|   `-- vn_traffic/          # perception, analytics, evidence, reasoning contracts
|-- tests/                   # pipeline, analytics, evidence, and reasoning tests
|-- app.py                   # Streamlit dashboard over a pipeline run directory
|-- detect.py                # backward-compatible root CLI
|-- run_pipeline.py          # repository-local MVP pipeline CLI
|-- environment.yml          # audited Conda environment
|-- pyproject.toml
`-- LICENSE                  # AGPL-3.0-only
```

Large datasets, model weights, source/generated videos, runtime outputs, and
video-specific configs/manifests are excluded from version control. A public
clone therefore contains no media. Supply local paths at runtime with
`--source <video_path>` and `--model <model_path>`; do not commit private paths
into shared YAML files.
