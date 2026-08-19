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

## Offline detection and tracking pipeline

The MVP path uses one YOLO instance for detection and ByteTrack. It writes the
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

# UAV moving-camera mode with global motion compensation
python run_pipeline.py --config configs/pipeline/offline_video_uav_gmc.yaml
```

The default config references the validation-selected v5 checkpoint. Override
`--model` to run another checkpoint without editing the YAML. Use
`configs/pipeline/offline_video_uav_gmc.yaml` for moving/panning UAV footage
(`analytics.mode: uav_motion`, `gmc_enabled: true`) instead of the
fixed-camera default.

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

## VLM/LLM reasoning (ad hoc)

There is no standalone CLI yet for the reasoning stage; it is driven by
`src/vn_traffic/reasoning/vlm_runtime.py` (`run_vlm_case`) and
`src/vn_traffic/reasoning/llm_runtime.py` (`run_llm_case`), given a frozen
event + evidence record from a pipeline run. See
[the reasoning protocol](reasoning_protocol.md) for the two-stage contract
and prompt versions.

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
