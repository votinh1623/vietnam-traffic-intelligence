# VLM/LLM reasoning protocol

## Status and split boundary

Reasoning evaluation input v1 is frozen at
`manifests/reasoning/evidence_eval_v1/input_lock.json`. It contains all 14
evidence records from H.264 acceptance run15: 13 line-crossing cases and one
congestion-transition case. The media remain local and ignored by Git; source,
manifest, event, decoded-frame, and encoded-artifact hashes bind the public
lock to the exact local inputs.

Run16 is excluded from evaluation and remains an independent candidate for
prompt development or quantization calibration. Its 146 evidence records are
now frozen separately at `manifests/reasoning/evidence_dev_v1/input_lock.json`
with lock SHA-256
`ba98c616d3f36882ea44ce0ded586aca1f0a984f34b1d421e533ba5158fbfe86`.
Evaluation cases must not be
used to select a model, prompt, precision, decoding parameter, threshold, or
runtime backend. If an evaluation case is inspected during development, v1 is
compromised and a new source-disjoint evaluation version is required.

The input lock remains `inputs_frozen_annotations_pending`. Both independent
review files now contain 14 complete records, but final adjudication is still
pending. This is enough to prevent input drift, but not enough to report
VLM/LLM quality. The 14 cases are an initial engineering gate, not a
statistically strong research test; publication claims require more
source-disjoint, human-annotated scenes.

## Two-stage contract

Contract version 1 keeps visual inference and report generation separate:

```text
frozen event + hashed media
            |
            v
VLM request -> VLM assessment (visual claims + evidence refs)
            |
            v
LLM request -> Vietnamese report (deterministic numbers + visual findings)
```

`src/vn_traffic/reasoning/contracts.py` builds and validates both boundaries.
Unknown fields are rejected so a model/backend cannot silently expand the
meaning of a versioned result.

### VLM request and assessment

The VLM receives the complete deterministic event and hashed keyframe/clip
references. It must not change deterministic fields, infer physical speed, or
infer event cause. Each visual observation contains Vietnamese text,
confidence in `[0, 1]`, and at least one known evidence reference. Incident
status is restricted to `observed`, `not_observed`, or `uncertain`; categories
are `none`, `collision`, `stalled_vehicle`, `road_hazard`, or `other`.

A line-crossing keyframe shows scene context but cannot by itself prove motion
direction or a crossing. Those values remain deterministic tracker outputs,
not VLM claims. A congestion clip supplies temporal context but still does not
prove the physical cause of congestion.

### LLM request and report

The LLM receives the validated VLM request and assessment. The report contains
a Vietnamese summary, traffic state, numeric facts, visual findings, action,
and limitations. Every numeric fact must provide an `event.*` source path and
exactly equal that source value. The validator rejects altered counts,
occupancy, timestamps, or pixel speeds. For congestion transitions,
`traffic_state` must equal `event.current_state`; otherwise it is
`UNSPECIFIED`.

Prompt wording is versioned in `configs/reasoning/prompts_v1.yaml`. Every
benchmark row must record its content hash, model and artifact hash, backend,
precision, generation parameters, contract version, input-lock hash, and Git
commit.

## Initial development candidates

`configs/reasoning/development_v1.yaml` records the RTX-host candidates:
`Qwen/Qwen3-VL-2B-Instruct` for visual assessment and Qwen3 text checkpoints
for Vietnamese report generation. They are upstream Qwen Apache-2.0
checkpoints; the text-model family advertises multilingual support across more
than 100 languages. The local Transformers 5.14.1 environment hosts the pinned
VLM and text artifacts.
The VLM repository contains
4,266,648,961 bytes across 12 hashed files. Its weight file SHA-256 is
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`;
the full local file manifest is
`manifests/models/qwen3-vl-2b-instruct.json`. The LLM repository contains
4,079,450,110 bytes across 12 hashed files; its manifest is
`manifests/models/qwen3-1.7b.json`.

Qwen3-1.7B FP16 failed before generation on this host with Windows error 1455
(`paging file is too small`) despite direct-GPU, low-CPU-memory loading. The
project therefore keeps that artifact as an unexecuted primary candidate and
uses pinned Qwen3-0.6B FP16 as the functional-demo fallback. The fallback
contains 1,519,209,243 bytes across 10 hashed files, recorded in
`manifests/models/qwen3-0.6b.json`. This is a host-memory compatibility choice,
not evidence that the smaller model has sufficient report quality.

The two models must load, run, and unload sequentially because concurrent
residency is not assumed to fit the RTX 3050 6 GB. FP16 is the reference path;
INT8/INT4 remains a later measured comparison. Before the first download, the
resolved immutable Hugging Face revision must replace `revision: null` and be
recorded with artifact hashes. Model IDs, expected parameter counts, or vendor
benchmarks are not local memory/latency evidence.

The first development case exposed a grounding failure: when the full event
JSON was placed in the visual prompt, the model copied `direction: down` into
an image claim and contradicted itself in limitations. Prompt-only warnings did
not remove the behavior. The runtime now gives the VLM only event identity
fields and withholds class, direction, measurements, and congestion state;
full deterministic data remains available to the later LLM boundary. A
keyframe-only grounding gate also rejects motion phrases.

After this correction, one FP16 keyframe smoke generated contract-valid,
static Vietnamese observations. The clean-commit recorded generation took
35.41 seconds and peaked at 5,683,444,224 allocated VRAM bytes. Its full
provenance is in `experiments/qwen3_vl_2b_dev_smoke_20260817/run.json`. This
single cold-process run only proves load/generate/validate feasibility. It is
not a latency distribution, quality result, or evidence that FP16 leaves
enough headroom for concurrent detector execution.

The LLM adapter accepts only a stored VLM result whose contract is marked
valid and whose case, event, citations, and assessment all revalidate against
the frozen request. The LLM generates only `summary_vi` and `action`; the
application assembles IDs, state, numeric facts, visual findings, and
limitations from authoritative inputs before validating the complete report.
The model therefore cannot introduce or alter a value under those fields. A
dry run verifies this boundary without loading weights:

```powershell
python scripts/reasoning/run_llm.py `
  --config configs/reasoning/development_v1.yaml `
  --case-id development-0001 `
  --vlm-result output/reasoning/dev_v1/development-0001-clean.json `
  --dry-run
```

The adapter and downloaded files alone are not an LLM quality or execution
result; that claim requires a contract-valid generated artifact from a clean
commit.

That clean-run functional demo is recorded in
`experiments/qwen3_vl_llm_dev_demo_20260818/run.json`. Qwen3-0.6B generated the
two prose fields in 14.87 seconds with 1,380,997,120 peak allocated VRAM bytes;
the application assembled the authoritative fields and the complete report
passed contract validation. Manual review still found generic wording and an
ambiguous reference to speed, so this is a functionality smoke only—not a
reasoning-quality result. The failed 1.7B load and two rejected 0.6B prompt
iterations are retained in the experiment record rather than hidden. The
"generic wording" here was later root-caused to a copyable example sentence
in the prompt itself; see "Prompt-copying bug: v1 to v3" below.

Primary model cards:

- <https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct>
- <https://huggingface.co/Qwen/Qwen3-1.7B>

## Prompt-copying bug: v1 to v3

Every VLM/LLM run recorded before this fix, on every case tried, reproduced
one example sentence from the prompt verbatim ("Quan sát thấy các phương
tiện trong khung hình.") instead of describing the image -- this is the
source of the "generic wording" noted in the functional-demo paragraph
above. `configs/reasoning/prompts_v1.yaml`'s user-turn JSON example used
that sentence as the `claim_vi` value, and the 2B model copied it as a free
answer. `_prompt_text()` in `src/vn_traffic/reasoning/vlm_runtime.py` was
changed to use a non-Vietnamese placeholder instead
(`tests/test_vlm_runtime.py::test_prompt_example_is_not_copyable_vietnamese_prose`).

That alone was insufficient: `prompts_v2.yaml`'s *system* prompt illustrated
the required structure with a different complete sentence ("chủ yếu là xe
máy, có nhiều ô tô con và vài xe buýt hoặc xe tải"), and the model copied
that one instead -- on a real truck-dominated highway keyframe where the
pipeline's own analytics recorded `car:17, motorcycle:1, truck:46`, the
model still answered "chủ yếu là xe máy" (mostly motorcycles)
(`output/reasoning/adhoc/run34-vlm-v2prompt.json`). Any complete,
grammatical example sentence anywhere in the prompt is copyable, regardless
of which prompt it lives in.

`prompts_v3.yaml` replaces every example with a fill-in-the-brackets
template (`"Phần lớn phương tiện là [LOẠI XE CHIẾM ĐA SỐ...], ngoài ra có
[...]; mật độ [...]."`) that is not itself a valid answer if copied
literally
(`tests/test_vlm_runtime.py::test_system_prompt_v3_has_no_copyable_example_sentence`).
Re-run on two real, distinct keyframes, it produced grounded, differing
answers matching the pipeline's own analytics: motorcycle-dominant on the
jam clip (`output/reasoning/adhoc/run32-vlm-v3prompt.json`, occupancy
0.709) versus truck-dominant on the highway clip
(`output/reasoning/adhoc/run34-vlm-v3prompt.json`). The raw VLM sentence is
sometimes mildly repetitive; the downstream LLM report cleans it into a
single sentence while keeping `numeric_facts` exactly equal to the
deterministic event measurements. This is not a formally re-frozen evidence
set or a measured quality result -- v1 and v2 stay unchanged for historical
hash reference, and these are local ad-hoc runs, not new experiment
records.

A related, still-open gap: `validate_grounding_policy` only forbids motion
claims when the VLM request has no clip evidence, but `run_vlm_case` never
actually loads or shows clip frames to the model -- it is keyframe-only
regardless of what the request references. A future motion claim on a
clip-bearing event would therefore not be caught as ungrounded even though
the model never saw the clip. It happened not to matter in the runs above
because the model made no motion claims, but the gap is real and unfixed.

## Annotation and metrics

Two independent pending templates and a local-media index are checked in under
`manifests/reasoning/evidence_eval_v1/annotations/`. Reviewer A and reviewer B
must work in separate files without reading the other reviewer's answers.

Each record uses annotation schema version 1:

| Field | Allowed value or rule |
|---|---|
| `annotation_status` | Keep `pending` while editing; set `complete` only after every field is reviewed |
| `evidence_quality` | `clear`, `partial`, or `poor` |
| `visible_density` | `low`, `medium`, `high`, or `uncertain`; visual image-plane judgment only |
| `visible_classes` | Unique values from pedestrian, car, motorcycle, bus, truck, other |
| `event_visual_support` | `supported`, `not_supported`, or `insufficient` |
| `incident_status` | `observed`, `not_observed`, or `uncertain` |
| `incident_category` | none, collision, stalled_vehicle, road_hazard, or other |
| `observations_vi` | One or more directly visible Vietnamese observations |
| `reference_summary_vi` | Conservative Vietnamese reference summary combining event and visible evidence |
| `required_limitations_vi` | One or more limitations the generated answer should disclose |
| `notes_vi` | Optional reviewer/adjudication note |

`not_observed` must use incident category `none`. Do not translate pixel speed
to km/h, infer a collision or congestion cause from vehicle density, or treat a
single line-crossing keyframe as proof of direction. `review_index.csv` maps
each case to its local keyframe/clip and deterministic event JSON.

The templates are created once with:

```powershell
python scripts/reasoning/manage_annotations.py prepare `
  --lock manifests/reasoning/evidence_eval_v1/input_lock.json `
  --output-dir manifests/reasoning/evidence_eval_v1/annotations `
  --reviewers reviewer_a reviewer_b
```

The command refuses to overwrite existing annotation files. After a reviewer
sets all 14 rows to `complete`, validate independently:

```powershell
python scripts/reasoning/manage_annotations.py validate `
  --lock manifests/reasoning/evidence_eval_v1/input_lock.json `
  --annotations manifests/reasoning/evidence_eval_v1/annotations/reviewer_a.jsonl `
  --require-complete
```

Repeat for reviewer B, then create—not resolve—the adjudication queue:

```powershell
python scripts/reasoning/manage_annotations.py compare `
  --lock manifests/reasoning/evidence_eval_v1/input_lock.json `
  --first manifests/reasoning/evidence_eval_v1/annotations/reviewer_a.jsonl `
  --second manifests/reasoning/evidence_eval_v1/annotations/reviewer_b.jsonl `
  --output manifests/reasoning/evidence_eval_v1/annotations/adjudication.json
```

Every case enters the queue because free-text references require human
adjudication even when categorical fields agree. The tool reports categorical
disagreements but never chooses a winner automatically.

The queue stores canonical SHA-256 values for both source annotation sets.
Validate that it still matches the current reviews with:

```powershell
python scripts/reasoning/manage_annotations.py validate-queue `
  --lock manifests/reasoning/evidence_eval_v1/input_lock.json `
  --first manifests/reasoning/evidence_eval_v1/annotations/reviewer_a.jsonl `
  --second manifests/reasoning/evidence_eval_v1/annotations/reviewer_b.jsonl `
  --queue manifests/reasoning/evidence_eval_v1/annotations/adjudication.json
```

If a reviewer legitimately changes an answer before adjudication begins,
rerun `compare` with `--replace-pending`. Replacement is allowed only while
every queue item is pending and contains no adjudicated result. A completed
queue must contain 14 schema-valid records from a third reviewer ID distinct
from reviewer A and reviewer B.

The current reviewer sets agree on every categorical field except
`evaluation-0004.event_visual_support`: reviewer A uses `supported`, while
reviewer B uses `insufficient`. The clip clearly supports a congested scene,
but does not show a distinct visual transition at frame 51; that distinction
must be resolved explicitly by the independent adjudicator.

Before model comparison, the two reviewers must annotate every case and an
adjudicator must resolve the queue. The review must distinguish:

- directly visible observations from deterministic event fields;
- supported, unsupported, and uncertain incident claims;
- sufficient versus insufficient evidence;
- acceptable Vietnamese summary content and prohibited causal claims.

Automatic gates are schema validity, event/case identity, known evidence
citation, finite confidence, exact numeric fidelity, and traffic-state
fidelity. Human-scored metrics are supported-claim precision, unsupported-
claim rate, incident accuracy where evidence is sufficient, summary
correctness, and limitation disclosure. Latency, memory, and artifact size are
reported separately from quality.

FP16, INT8, and INT4 candidates must use the same frozen inputs and annotations.
A quantized model passes only if it satisfies the agreed quality tolerance
against the reference model; smaller size or higher throughput alone is not a
quality result.

## Rebuild and verification

The checked-in lock can be reproduced while run15 and its source media are
available:

```powershell
python scripts/reasoning/freeze_evidence_set.py `
  --run-dir output/pipeline/run15 `
  --set-id vietnam_traffic_reasoning_eval_v1 `
  --split evaluation `
  --output manifests/reasoning/evidence_eval_v1/input_lock.json
```

The command verifies the source-video hash and every keyframe/clip hash before
writing atomically. Rebuilding identical inputs produces the same
`lock_sha256`; any event, evidence record, or artifact change fails validation
or produces a different lock.
