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

