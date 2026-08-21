# Paper outline (IMRaD)

Draft outline only -- skeleton for a future manuscript, not a written paper.
Every claim below must be checked against the actual current numbers in
[the readme](../readme.md) and [benchmark protocol](benchmark_protocol.md)
before drafting prose; this file may drift out of date as experiments
continue. Working title and framing choices are marked `[decide]` where the
project itself hasn't settled the question yet.

## Working title

`[decide]` Candidate framing (pick one before drafting):
- *"Why a Validation-Set Gain Disappeared: A Disciplined Small-Object
  Detection Case Study on VisDrone"* -- leads with the methodology finding,
  honest about the modest results.
- *"Toward Reliable Small-Object Detection and Tracking for UAV Traffic
  Monitoring: A Leakage-Controlled Evaluation Case Study"* -- leads with the
  application, methodology as a secondary contribution.

The project's own strongest asset so far is measurement discipline, not
result magnitude (see Discussion) -- the title should not oversell.

---

## I. Introduction

1. **Motivation / importance**
   - UAV traffic monitoring in dense scenes dominated by small vehicles and
     pedestrians (motorcycles especially) is practically important and
     technically hard: small object size, heavy occlusion, severe scale
     variation (cite VisDrone benchmark paper, small-object survey -- see
     [literature review](literature_review.md)).
   - A useful system needs correct detection *and* tracking *and* honest
     accuracy reporting, not just a demo that looks convincing.
2. **Research objectives** (state precisely, matching the project's actual
   [Research objectives](../readme.md#research-objectives) table):
   - Improve small-object detection accuracy on a real aerial benchmark.
   - Determine whether the same improvement propagates into tracking
     without changing the tracking algorithm.
   - Demonstrate and validate a measurement protocol that avoids
     overstating a result inflated by repeated selection against the same
     evaluation split.
3. **Scope statement**
   - VisDrone2019-DET/MOT only, vehicle classes (the project's actual
     operational scope); pedestrian/people classes measured but out of
     product scope.
   - Explicitly *not* claiming: a novel architecture, a state-of-the-art
     leaderboard result, or a validated Vietnam-domain deployment (the
     project's original Vietnam v5 dataset is retained only as a
     qualitative field check, see [Dataset](../readme.md#dataset)).
4. **Organization of the paper** (standard IMRaD signpost paragraph).
5. **Literature review** -- pointer to
   [docs/literature_review.md](literature_review.md); paper draft should
   fold the relevant entries (VisDrone, NWD, FixRes, Copy-Paste, SAHI,
   ByteTrack/BoT-SORT, HOTA, Dwork et al., Recht et al.) into inline
   citations here rather than a separate section, per IMRaD convention.

## II. Methods

1. **Dataset and splits**
   - VisDrone2019-DET (train/val/test-dev) and VisDrone2019-MOT (val only);
     `test-challenge` excluded (ground truth not public).
   - The locked-test discipline itself as a method: test-dev sourced and
     frozen 2026-08-21, never used for selection, read once per candidate.
   - `[decide]` Whether to include the Vietnam v5 leakage-repair story as a
     methods appendix (it motivated the discipline applied to VisDrone) or
     omit it entirely as out of scope.
2. **Detector**
   - YOLOv8s, COCO -> VisDrone2019-DET initialization; why YOLOv8 (see
     [benchmark protocol](benchmark_protocol.md#why-yolov8)).
   - Evaluation protocol: COCO-style AP/AP-small via a frozen pycocotools
     evaluator, `imgsz` modes compared (640, 1280, SAHI tiles, hybrid).
3. **Detection-side interventions tested** (describe method only, not
   results -- results belong in section III):
   - NWD bbox loss (Wasserstein-distance-based box similarity).
   - P2 detection head (added stride-4 output scale).
   - Native high-resolution continuation fine-tune (train/infer resolution
     matching).
   - Scale-aware copy-paste augmentation (vehicle-class crop bank, 8-28px
     target scale at 1280 letterbox, IoU-gated placement).
4. **Tracker**
   - ByteTrack baseline; BoT-SORT ablation; ReID ablation (`model:auto`
     feature reuse vs. a real pretrained embedding).
   - HOTA/DetA/AssA via TrackEval, plus classic CLEAR-MOT metrics via a
     repaired local motmetrics evaluator.
5. **Reproducibility framework**
   - Hash-backed run manifests (config/weights/data SHA-256, git commit)
     for every experiment; pre-registered gates committed to version
     control *before* running a candidate, not after.
   - `[decide]` Whether to formalize this as a named contribution
     ("gate-then-run protocol") or present it only as methodological detail.

## III. Results

Report each intervention's outcome plainly, including negative ones --
this is where the paper's honesty is demonstrated, not asserted.

1. **Detector baseline** on VisDrone-DET-val and (once available for every
   candidate) test-dev.
2. **NWD ablation** -- rejected, worse on every metric; report per-class
   breakdown.
3. **P2 ablation** -- rejected, worse than both baseline and NWD; disclose
   the training-incident confounds (batch size, restarted schedule) rather
   than hiding them.
4. **Highres native-resolution pilot**
   - Passed its pre-registered gate on val (report the val numbers).
   - **Central result**: did not clearly replicate on the locked test-dev
     read (report both val and test-dev numbers side by side, plus the
     vehicle-only recomputation that ruled out class-averaging as the
     cause).
   - Tracking propagation check (val-based, same caveat applies).
5. **ReID ablation** (pretrained embedding vs. feature reuse) -- negative,
   consistent with the DetA/AssA decomposition.
6. **Copy-paste augmentation pilot** -- `[pending]`: training was
   interrupted before completion (2026-08-21); this section cannot be
   written until the pilot actually runs and is read against test-dev.
   Do not draft placeholder numbers.
7. **Diagnostic: why the gain didn't replicate** -- the val-vs-test-dev
   ground-truth distribution comparison (box-scale by class, vehicle-only
   recomputation), reported as an open finding without a fully isolated
   root cause.
8. **Secondary results** (counting, alerts/congestion, GMC correction,
   VLM/LLM description) -- likely condensed into a shorter subsection or
   supplementary material, since they are not the paper's central claim.

## IV. Discussion

1. **What the results mean.** Four independent levers targeting
   small-object detection accuracy (NWD, P2, highres-native-resolution,
   [pending: copy-paste]) have not yet produced a result confirmed to
   survive a genuinely held-out test. State this plainly; do not spin it.
2. **The methodological finding is the paper's actual contribution
   candidate.** Position the val-to-test-dev gap finding against the two
   literature framings that only partially reconcile with it (Dwork et
   al.'s adaptive-data-analysis overfitting vs. Recht et al.'s
   harder-distribution-without-overfitting) -- state explicitly that this
   project's evidence (the *baseline* checkpoint's own AP-small also drops
   substantially val-to-test-dev) leans toward Recht's framing, while the
   *relative* gain shrinking so much more than Recht's "no diminishing
   returns" finding predicts leans toward Dwork's. Neither fully explains
   the data; say so.
3. **Why detection quality, not tracking algorithm choice, is the
   bottleneck.** DetA/AssA decomposition and the ReID-embedding negative
   result both point the same direction -- discuss what this implies for
   where future engineering effort should go (detection-side levers over
   tracker-side ones).
4. **Limitations**
   - Single dataset family (VisDrone); Vietnam-domain generalization
     unvalidated (demoted to field-validation-only, see
     [Field validation](../readme.md#field-validation-vietnam-clips-historical)).
   - No locked test yet for tracking/ReID (MOT-val only) -- the same
     inflation risk is plausible there and unmeasured.
   - Consumer GPU (RTX 3050, 6GB) constrains experiment scale (batch size,
     epoch count, no large-scale hyperparameter search) -- some rejected
     ablations carry real confounds (P2's batch=2 vs. 4, restarted
     schedule) that a larger GPU budget would have let the project isolate
     cleanly.
   - No comparison yet to published VisDrone leaderboard numbers to
     calibrate how far from achievable performance this project's numbers
     sit.
5. **Threats to validity** -- the "no more than one test-dev read per
   candidate" rule is a blunt heuristic, not a formally validated
   correction (contrast with Dwork et al.'s `Thresholdout`); if many
   candidates are read against test-dev over the project's remaining
   lifetime, cumulative inflation risk returns even under this rule.

## V. Conclusion

1. Restate objectives from the Introduction.
2. **Main findings**, ranked by confidence:
   - High confidence: detection recall, not tracking algorithm choice, is
     this pipeline's bottleneck (DetA/AssA decomposition, ReID ablation).
   - High confidence: repeated selection against a small, fixed validation
     split measurably inflates an apparent small-object detection gain
     (empirically demonstrated, not just theorized).
   - Low/no confidence yet: any specific detection-side lever
     (loss/architecture/resolution/augmentation) reliably improves
     small-object AP on genuinely unseen VisDrone data. `[decide]` This
     conclusion may change once the copy-paste pilot completes -- do not
     finalize this section before that.
3. **Significance.** A disciplined, reproducible protocol for small-object
   detection research under compute constraints, with a concrete
   demonstration of why held-out discipline matters -- offered as
   something other small-scale UAV/aerial-CV projects can reuse, separate
   from whether any specific accuracy lever proposed here works.
4. **Future work**
   - Complete and gate the copy-paste pilot.
   - Retest SAHI with slicing-aware *fine-tuning* (not inference-only), per
     the SAHI paper's own better-performing configuration.
   - Compare BoT-SORT's built-in GMC against this project's custom ECC
     implementation.
   - Lock an equivalent held-out split for tracking/ReID.
   - Revisit NWD with a train-distribution-scaled constant before treating
     it as fully closed.
   - Only after a detection lever is confirmed: re-evaluate Vietnam-domain
     applicability as a distinct, separately-scoped study.

---

## Appendix / supplementary material candidates

- Full experiment history: [docs/history.md](history.md).
- Full method/rationale detail: [benchmark protocol](benchmark_protocol.md).
- Dataset construction and integrity protocol:
  [dataset protocol](dataset_protocol.md).
- Reproducibility manifests: `experiments/*/run.json` (hash-backed, one per
  experiment referenced in Results).
