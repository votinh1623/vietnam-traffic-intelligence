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

Before model comparison, two reviewers should independently annotate each
case, then adjudicate disagreements. The annotation guide must distinguish:

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
