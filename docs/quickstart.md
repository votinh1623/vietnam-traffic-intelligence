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

## Detection (single image/video CLI)

`--model` defaults to `runs/detect/research/yolov8s_v5_seed0/weights/best.pt`
(the validation-selected v5 checkpoint, see `scripts/detect.py`) -- pass
`--model` explicitly to run a different checkpoint.

```powershell
# One image -- uses the default checkpoint above
python detect.py test_image.jpg --conf 0.5

# Directory of images
python detect.py datasets/vn_images --conf 0.4

# Video
python detect.py datasets/raw_videos/traffic_jam.mp4 --conf 0.3

# To use a different checkpoint, pass --model explicitly:
python detect.py test_image.jpg --model runs/detect/baseline/yolov8s_visdrone/weights/best.pt
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
# Validate paths without loading the model
python run_pipeline.py --dry-run

# Run after GPU training is complete
python run_pipeline.py `
  --source datasets/raw_videos/traffic_normal.mp4 `
  --model runs/detect/research/yolov8s_v5_seed0/weights/best.pt

# Short integration check
python run_pipeline.py --max-frames 30 --imgsz 640

# The two WP1 demo videos (BoT-SORT+ReID, measurement manifest, multi-view
# evidence) -- see reports/ke-hoach-pipeline-va-mo-phong.md Gate G1
python run_pipeline.py --config configs/pipeline/offline_video_0681.yaml
python run_pipeline.py --config configs/pipeline/offline_video_7938.yaml

# Visual heatmap for severely occluded jams the detector draws no boxes over
python run_pipeline.py --config configs/pipeline/offline_video_stillness_heatmap_demo.yaml
```

The default config references the validation-selected v5 checkpoint. Override
`--model` to run another checkpoint without editing the YAML.
`analytics.mode: uav_motion` + `gmc_enabled: true` (global motion
compensation for a moving/panning camera) is still a supported config
option; the demo config that exercised it (`offline_video_uav_gmc.yaml`)
was retired along with the highres-pilot checkpoint it depended on. The
current UAV demo configs (`offline_video_0681.yaml`, `offline_video_7938.yaml`)
use the fixed-camera default instead.

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

`scripts/reasoning/run_vlm.py` and `scripts/reasoning/run_llm.py` are
standalone CLIs over one frozen development case (event + evidence record
from a completed pipeline run, frozen via
`scripts/reasoning/freeze_evidence_set.py`). Both need a local model
directory (`models/qwen3-vl-2b-instruct`, `models/qwen3-0.6b`) -- weights
are not downloaded automatically. `--dry-run` validates the case (evidence
hashes, model IDs) without loading either model.

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
applications and retry). See [the reasoning protocol](reasoning_protocol.md)
for the two-stage contract and prompt versions.

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

Large datasets, model weights, generated videos, and runtime outputs are kept
outside version control. The local workspace retains the current v5 dataset
and the single current detector checkpoint
(`runs/detect/research/yolov8s_v5_seed0/`). Earlier research directions
(NWD loss, P2 detection head, copy-paste augmentation, highres fine-tune,
TVLR, SAHI, detector/tracker benchmark sweeps) were tried, measured, and
closed -- their code, configs, checkpoints, and raw evaluation outputs have
been removed rather than kept as unused weight. `output/pipeline/run16` is
kept as the artifact_root for the committed default reasoning lock
(`manifests/reasoning/evidence_dev_v1/input_lock.json`); `run85`/`run86` are
the current WP1 demo pipeline outputs
(`configs/reasoning/wp1_demo_7938.yaml`/`wp1_demo_0681.yaml`). Other
temporary extraction sets, superseded smoke/finetune runs, and legacy
pipeline run outputs are intentionally removed once no longer needed.
