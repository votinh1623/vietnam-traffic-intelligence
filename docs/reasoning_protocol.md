# VLM/LLM reasoning protocol

## Status and split boundary

Reasoning evaluation input v1 is frozen at
`manifests/reasoning/evidence_eval_v1/input_lock.json`. It contains all 14
evidence records from H.264 acceptance run15: 13 line-crossing cases and one
congestion-transition case. The media remain local and ignored by Git; source,
manifest, event, decoded-frame, and encoded-artifact hashes bind the public
lock to the exact local inputs.

Run16 is excluded from evaluation and remains an independent candidate for
prompt development or quantization calibration. Evaluation cases must not be
used to select a model, prompt, precision, decoding parameter, threshold, or
runtime backend. If an evaluation case is inspected during development, v1 is
compromised and a new source-disjoint evaluation version is required.

The current lock status is `inputs_frozen_annotations_pending`. This is enough
to prevent input drift, but not enough to report VLM/LLM quality. The 14 cases
are an initial engineering gate, not a statistically strong research test;
publication claims require more source-disjoint, human-annotated scenes.

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
