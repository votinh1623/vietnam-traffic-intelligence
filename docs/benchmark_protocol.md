# Benchmark protocol

All result rows must identify the model hash, dataset/split manifest hash,
configuration, backend, precision, seed, software environment, hardware, and
Git commit. Unexecuted measurements are recorded as `TBD`; invalidated results
remain traceable with status `invalid` and a reason.

## Runtime measurement

- use batch 1 for the real-time path and report batched throughput separately;
- perform 20–50 warm-up iterations and at least 300 timed frames;
- repeat the run three times and report sample count, mean, p50, and p95;
- synchronize CUDA around timed GPU work;
- separate cold start from warm steady-state execution;
- time input/decode, preprocess, transfer, inference, postprocess, tracking,
  analytics, evidence building, reasoning, and end-to-end latency;
- record scene density/candidate count, peak RAM/VRAM, and thermal state;
- never infer power or energy from latency.

## Quality gates

A quantized artifact is benchmarkable only after it:

1. loads successfully in the declared backend;
2. produces finite outputs with the expected shapes;
3. passes representative output parity checks;
4. is evaluated on the same locked quality set as its reference precision.

Detector quality uses mAP50, mAP50-95, per-class metrics, small-object AP, and
motorcycle AP. Tracking uses HOTA, DetA, AssA, IDF1, MOTA, ID switches, and
fragmentation. LLM/VLM quality uses a task-specific frozen evaluation set,
numeric fidelity, and unsupported-claim rate.

Reasoning input v1 and its no-tuning boundary are defined in
`docs/reasoning_protocol.md`. Schema validity, evidence citations, numeric
fidelity, and traffic-state fidelity are automatic gates. Human annotations
are still required before supported-claim precision, incident accuracy,
summary correctness, or quantization parity can be reported.

## Tracking evaluation status

The local motmetrics evaluator is valid for CLEAR MOT and identity metrics
after repairing its IoU-distance construction. It uses `1-IoU` directly,
converts a minimum IoU threshold `t` to the motmetrics maximum distance
`1-t`, includes the union of GT and prediction frame indices, and aggregates
sequences with a combined OVERALL accumulator.

HOTA, DetA, and AssA are not provided by motmetrics. They remain `TBD` until
TrackEval is integrated and verified on a synthetic fixture. Historical root
tracking CSV files predate the repair and remain `invalid`.
