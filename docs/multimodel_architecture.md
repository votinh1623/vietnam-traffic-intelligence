# Multi-model deployment-readiness architecture

## Scope

The research system contains two independently measurable model workloads:

1. a continuous CV perception path for detection, tracking, and deterministic
   traffic analytics;
2. an event-triggered LLM/VLM reasoning path for summaries, explanations, and
   constrained visual inspection.

The project does not claim deployment on edge hardware until it is measured on
a real target. RTX results are reported as deployment-readiness evidence.

## Data flow

```text
video -> detector -> tracker -> deterministic analytics -> structured events
           |                                               |
           +---- hashed raw keyframes / event clips -------+
                                                           v
                                            prompt and evidence builder
                                                           v
                                            quantized LLM or VLM runtime
                                                           v
                                          schema-validated explanation
```

The detector remains the source of numeric counts and trajectories. The
reasoning model cannot invent measurements, infer an accident without an
explicit evidence policy, or override safety-critical deterministic rules.

## Implemented evidence boundary

The current Stage 3 implementation stops before model inference. A configurable
offline selector links each eligible deterministic event to a raw source
keyframe and, for congestion transitions, a bounded pre/post clip. The
`evidence.jsonl` manifest records source frame/time, relative paths, dimensions,
clip bounds, the source-video hash, artifact hashes, and the pre-encoding BGR
frame hash. A single second pass decodes the processed span sequentially and
feeds all keyframes and overlapping clip writers without random frame seeking.
This makes the future prompt/VLM input frozen and auditable without treating
visual selection as a semantic conclusion.

Line crossings receive keyframes only by default to avoid producing many nearly
redundant clips. No caption, incident label, severity judgment, or natural-
language claim is generated at this stage.

## RTX 3050 scheduling constraint

Six GB of VRAM is not enough to assume that a 1280-pixel detector and a useful
VLM can remain resident concurrently. The default design therefore gives the
detector priority and invokes the reasoning model on events/keyframes. The
benchmark must compare at least:

- detector-only continuous execution;
- sequential model loading/execution;
- concurrent residency only when it fits without paging or OOM;
- end-to-end event-to-report latency, including model load and image encoding.

## Quantization studies

### Perception model

PyTorch CUDA FP32/FP16, ONNX Runtime CUDA, TensorRT FP16, and TensorRT INT8 are
the planned RTX matrix. INT8 requires representative traffic calibration and
output/mAP parity checks before latency is accepted.

### Reasoning model

The LLM/VLM model and runtime are deliberately TBD. Candidate precisions are
FP16, INT8, and weight-only INT4. Quantization must be evaluated against the
same frozen prompt/evidence set and not only by model size or tokens/second.

LLM metrics include numeric fidelity to event JSON, unsupported-claim rate,
task accuracy, time to first token, generation throughput, peak RAM/VRAM,
cold start, and artifact size. VLM evaluation additionally requires a frozen
traffic visual-question/description set and image-conditioned correctness.

## Research claim boundary

Until a labeled reasoning benchmark exists, LLM/VLM integration is an
engineering contribution and its quality fields remain `TBD`. A fluent demo is
not evidence that quantization preserves reasoning quality.
