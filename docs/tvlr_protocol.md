# TVLR feasibility protocol (frozen before oracle evaluation)

## Research question and scope

This study asks whether post-NMS, low-confidence detector proposals can recover
vehicle observations missed by the selected ByteTrack pipeline when those
proposals are verified by temporal evidence. The intended method is offline:
forward/backward support uses future frames and is not claimed to support live
alerts or causal real-time inference.

The study does **not** treat all low-confidence detections as recovered objects.
It first measures an oracle upper bound. A later TVLR implementation is useful
only if it improves over both naive confidence lowering and ByteTrack's built-in
second association stage.

## Distinction from ByteTrack

The selected ByteTrack profile already associates detections at or above
`track_low_thresh=0.1` to existing tracks in a second matching stage. TVLR is
therefore restricted to ground-truth observations satisfying both conditions:

1. no class-consistent ByteTrack output overlaps the observation at IoU 0.5;
2. the low-confidence detector export still contains a class-consistent
   post-NMS proposal at IoU 0.5.

The oracle reports these incremental candidates by score band:

- below ByteTrack's low threshold (`0.02 <= score < 0.1`);
- inside ByteTrack's association range (`0.1 <= score < 0.5`);
- at or above its high threshold (`score >= 0.5`).

The second and third bands expose observations that ByteTrack missed despite
receiving a nominally eligible detection. The first band is the clearest
non-duplicative TVLR opportunity. A future method must additionally require
one-sided or forward/backward support from the same tubelet; the oracle uses GT
identity only to measure this upper bound, never as method input.

## Data split

The locally available VisDrone2019-MOT-val sequences are frozen into a small
feasibility-development split and an untouched internal holdout in
`configs/evaluation/tvlr_oracle_visdrone_mot_v1.yaml`. The basketball-court
sequence is excluded from the traffic-counting claim. Only the development
split may be used for the Stage B go/no-go decision. The holdout must remain
unread until an implementation and its thresholds are frozen.

This is an internal feasibility protocol, not an official VisDrone test. Before
a paper experiment, VisDrone MOT train must be obtained and used for method
development; the official validation/test protocol and a new Vietnam-source
holdout must then be frozen independently.

## Export controls

The detector is run once at confidence 0.02 and its post-NMS boxes are cached.
The export records the number of frames reaching `max_det=3000`. Any saturated
frame invalidates the assumption that the cache covers available proposals and
requires a controlled max-det sensitivity check. Runtime and peak CUDA memory
are recorded; the 6--10 minute estimate is not an acceptance criterion.

## Locked Stage B gates

TVLR proceeds to implementation only when the development split satisfies:

- at least 15% of ByteTrack-missed tiny/occluded GT observations have a
  class-consistent low-confidence proposal; and
- either oracle detection recall improves by at least 0.02 absolute, or oracle
  frame-count WAPE improves by at least 5% relative; and
- a non-zero subset of the incremental candidates has temporal support. The
  report must separate one-sided support from strict forward/backward support.

These are engineering go/no-go gates, not post-hoc paper success criteria. If
they fail, TVLR stops before implementation or training. If they pass, Stage C
must still beat naive low-confidence inference and ByteTrack low-score
association without using GT identity.

## Claim boundary

Oracle recovery is an upper bound because it uses GT class, box and identity to
select true proposals without admitting false positives. It is not a model
result, must not be placed in the main results table as achieved performance,
and cannot establish real-time capability.
