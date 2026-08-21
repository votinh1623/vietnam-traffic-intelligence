# Dataset integrity protocol

**Status (2026-08-21): VisDrone2019-DET/MOT is now the primary dataset** for
detector/tracker development and evaluation (see the readme's
[Dataset](../readme.md#dataset) section). Only VisDrone `train`/`val` are
available locally; `test-dev` (GT public) needs to be sourced and placed at
`datasets/VisDrone/VisDrone2019-DET-test-dev/` to close the "selecting
against val repeatedly" gap disclosed there. Everything below this point
documents the Vietnam dataset work (`v2` through `v5`), retained as
historical record and field-validation source only -- it is not the active
benchmark.

The current `vietnam_dataset_v2` split is retained as a legacy development
artifact. It is not a valid scientific test because source videos and adjacent
frames occur across train, validation, and test.

## Split unit

The minimum grouping unit is a source video. If multiple videos share a
camera, location, recording session, or re-uploaded segment, the grouping unit
must be raised to that shared identity. No group may occur in more than one of
train, calibration, validation, and test.

Newly collected sources should be reserved for the locked test set. The test
manifest is frozen before model selection and must not be used to select input
resolution, confidence thresholds, tracker parameters, prompts, or
quantization settings.

## Calibration

Quantization calibration is independent of validation and test. With limited
data it may be sampled from the training partition, but its images must be
excluded from gradient updates for the corresponding experiment and recorded
in a calibration manifest. A source-independent calibration partition is
preferred when enough videos are available.

Detector and VLM/LLM calibration are separate artifacts:

- detector INT8 calibration represents Vietnamese traffic images and uses the
  exact deployment preprocessing;
- LLM/VLM quantization calibration represents the prompts, event JSON, text,
  and sampled images used by the reasoning workload;
- neither calibration set is an evaluation set.

## Manifest requirements

Each image record should include source ID, frame index/timestamp, split,
label status, class counts, content hash, and—when available—URL, city,
camera/location, session, license, and annotation QA state.

Raw web videos, extracted frames, labels, and model weights are not committed
to the public repository unless redistribution rights are documented.

## Audit command

```powershell
python scripts/data/audit_dataset.py `
  --dataset-root datasets/vietnam_dataset_v2 `
  --output-dir manifests/datasets/vietnam_v2_legacy
```

The generated source split is a proposal for human review. The script never
moves or edits dataset files.

## Vietnam dataset v4

`vietnam_dataset_v4` is materialized from the legacy export without modifying
it. Sources are assigned exclusively to train, calibration, validation, or
test. Polygon annotations are deterministically converted to bounding boxes.

Frames with conflicting duplicate annotations are excluded rather than
silently selecting one label version. The generated `exclusions.csv` is the
review queue. The four generic `frame_*.jpg` records are also excluded because
their source cannot be established.

The renamed YouTube clips (`traffic_jam`, `traffic_normal`, `vid3`, and
`vid4`) were checked against all other source groups with a 16x16 difference
hash. No candidate overlap was found at Hamming distance 12; the closest pair
was distance 86. This supports treating them as separate sources, while the
report remains a candidate audit rather than proof of licensing or provenance.

The v4 test split was locked on 2026-08-17. Its content-addressed manifest is
`manifests/datasets/vietnam_v4/test_lock.json`, with lock SHA-256
`9c65a74d1a75ce6619d9e12396ac5f2e690514ff5b632bae6c28105377d964c5`.

## Vietnam dataset v5

The first v4 smoke run showed that Ultralytics was removing exact duplicate
boxes at load time. V5 supersedes v4 for research runs: it applies the same
source split and polygon-to-box conversion, then removes 53 exact duplicate
boxes during materialization. No duplicate boxes remain in any split. V5 has
its own locked test manifest and must not be compared as though it were the
same dataset version as v4.

### Source composition and the object-scale gap

All 1,214 v5 images come from only 12 source videos (5 train / 2 calibration
/ 2 validation / 3 test). 11 are repurposed YouTube uploads (7 auto-named on
download, plus `traffic_jam`, `traffic_normal`, `vid3`, and `vid4`, confirmed
as YouTube sources); exactly 1 (`DJI_20250516071323_0341_D`, in validation)
is a native drone capture. The two sources explicitly titled as aerial drone
footage both landed in the locked test by the source-disjoint split -- with
only 12 sources total, this is close to unavoidable, not a labeling defect.

Measured directly from the label files (`sqrt(w*h)` in pixels at
imgsz=1280): only 4.7% of train boxes are under 16px versus 48.8% of test
boxes. A model trained on this data sees real small-object examples rarely,
then is evaluated on a split where they dominate. This is the likely root
cause of the detector's validation-to-test gap (see
[benchmark protocol](benchmark_protocol.md)): a data scarcity problem, not
something a loss function or architecture change alone can fully correct --
both an NWD bbox-loss ablation and a P2 architecture ablation targeting this
exact gap were tried and rejected.
