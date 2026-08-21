# Literature review

Papers behind the technologies, algorithms, and problems this project has run
into in practice, found via web search 2026-08-21. Grouped by the part of the
system/process each connects to. This is a reading list with the specific
relevance noted, not a survey written from scratch -- read the source before
citing any specific number.

## Dataset and problem framing

- **Zhu et al., "Vision Meets Drones: A Challenge"** (VisDrone benchmark,
  [arXiv:1804.07437](https://arxiv.org/pdf/1804.07437)). The source of
  VisDrone2019-DET/MOT, now this project's primary dataset. 14 Chinese
  cities, four tracks (image detection, video detection, single-object
  tracking, multi-object tracking); explicitly frames small object size,
  heavy occlusion, variable density, and severe scale variation as the
  benchmark's core challenges -- exactly the failure modes this project has
  been chasing (the object-scale gap, the severe-occlusion congestion
  blind spot).
- **"Small Object Detection: A Comprehensive Survey on Challenges,..."**
  ([arXiv:2503.20516](https://arxiv.org/pdf/2503.20516)). General survey:
  multi-scale feature extraction, attention mechanisms, transformer
  architectures, super-resolution, and data augmentation/synthetic data as
  the main lever families. Useful as an index of what has *not* been tried
  yet here (attention/transformer heads, super-resolution preprocessing) --
  this project has only tried the loss-side (NWD), architecture-side (P2),
  resolution-side, and augmentation-side (copy-paste) levers.

## Detection: loss and architecture (rejected on Vietnam v5)

- **Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny
  Object Detection"** ([arXiv:2110.13389](https://arxiv.org/abs/2110.13389),
  [code](https://github.com/jwwangchn/NWD)). The basis of this project's
  rejected NWD ablation. Models boxes as 2D Gaussians and uses Wasserstein
  distance instead of IoU so similarity stays meaningful even at
  near-zero overlap -- the paper reports +6.7 AP over a fine-tuning
  baseline on their own AI-TOD benchmark. This project's NWD attempt was
  worse on every metric; the paper's own constant is tuned to AI-TOD's box
  sizes, and this project's post-hoc hypothesis (constant mismatched to
  train-batch box sizes, saturating the similarity term) is worth checking
  against the paper's own constant-selection guidance directly, which
  wasn't done before rejecting.
- Companion/extension: **"Detecting tiny objects in aerial images: a
  normalized Wasserstein distance and a new benchmark"**
  ([arXiv:2206.13996](https://arxiv.org/abs/2206.13996)) -- the journal
  version with the AI-TOD-v2 benchmark; worth checking for a
  train-distribution-scaled constant recipe before retrying NWD.

## Detection: train/test resolution mismatch

- **Touvron, Vedaldi, Douze, Jégou, "Fixing the train-test resolution
  discrepancy"** (FixRes, NeurIPS 2019,
  [arXiv:1906.06423](https://arxiv.org/abs/1906.06423)). Directly the
  phenomenon this project's highres pilot targeted: train/test-time object
  scale mismatch from decoupled train/inference resolutions, fixed by a
  cheap fine-tune at the target resolution. Important nuance the project
  hasn't engaged with: FixRes's own finding is that a *lower* train
  resolution plus test-time fine-tuning at higher resolution wins --
  opposite direction from assuming higher native training resolution is
  strictly better. Worth re-reading before deciding how many epochs /
  what learning rate a second highres attempt should use.
- Follow-up: **"Fixing the train-test resolution discrepancy:
  FixEfficientNet"** ([arXiv:2003.08237](https://arxiv.org/abs/2003.08237)).

## Detection: data augmentation

- **Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method
  for Instance Segmentation"** (CVPR 2021,
  [arXiv:2012.07177](https://arxiv.org/abs/2012.07177)). The basis for
  `scripts/train/copy_paste_augment.py`. Key finding directly relevant to
  this project's implementation choice: the paper found *purely random*
  placement (no scene/surface-aware placement, no blending) matches or
  beats more "realistic" placement heuristics -- supports keeping this
  project's simple hard-paste, random-position approach rather than adding
  complexity for realism. Not yet checked: their large-scale-jitter
  augmentation, used alongside copy-paste in the paper, which this
  project's pilot does not include (`mosaic=0.0`, no extra scale jitter
  beyond `scale: 0.2`).
- **SAHI: Akyon, Altinuc, Temizel, "Slicing Aided Hyper Inference and
  Fine-tuning for Small Object Detection"**
  ([arXiv:2202.06934](https://arxiv.org/abs/2202.06934)). Basis for this
  project's already-rejected SAHI/hybrid inference modes. Notably: the
  paper's own headline results are specifically on VisDrone and xView, and
  report large gains (+12.7% to +14.5% AP with slicing-aided *fine-tuning*,
  not just slicing inference) -- this project only tested slicing at
  *inference time* without the paper's fine-tuning step, which may explain
  why this project's SAHI ablation underperformed standard 1280 while the
  paper reports a win. A slicing-aware fine-tune (train on tiles, not just
  infer on tiles) is an untested variant, distinct from both the resolution
  fix and copy-paste.

## Tracking

- **Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box"** (ECCV 2022,
  [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)). The project's
  tracker baseline and pipeline default (`bytetrack_custom.yaml`).
- **Aharon, Orfaig, Bobrovsky, "BoT-SORT: Robust Associations
  Multi-Pedestrian Tracking"**
  ([arXiv:2206.14651](https://arxiv.org/abs/2206.14651)). This project's
  BoT-SORT/ReID ablation. Notable overlap not yet cross-checked: BoT-SORT
  already integrates its own camera motion compensation (its
  `gmc_method: sparseOptFlow` setting, present in this project's
  `botsort_custom.yaml`) -- separate from and never compared against this
  project's own hand-built ECC-based GMC
  (`src/vn_traffic/analytics/motion.py`), which only applies under
  ByteTrack. Worth checking whether BoT-SORT's built-in GMC on this
  project's UAV clips beats the custom implementation, instead of only
  comparing BoT-SORT vs. ByteTrack with ReID as the sole axis.
- **Luiten et al., "HOTA: A Higher Order Metric for Evaluating
  Multi-object Tracking"** (IJCV 2020,
  [PDF](https://www.cvlibs.net/publications/Luiten2020IJCV.pdf),
  [TrackEval code](https://github.com/JonathonLuiten/TrackEval)). The
  metric and reference implementation behind this project's
  `scripts/evaluate_hota.py` integration and the DetA/AssA decomposition
  that identified detection recall (not association) as the tracking
  bottleneck -- the single most load-bearing external result this project
  has used to prioritize work.

## Why the val-based gain didn't replicate on test-dev

- **Dwork, Feldman, Hardt, Pitassi, Reingold, Roth, "Generalization in
  Adaptive Data Analysis and Holdout Reuse"** (NeurIPS 2015,
  [arXiv:1506.02629](https://arxiv.org/abs/1506.02629)). The formal
  version of exactly the risk this project flagged and then confirmed
  empirically 2026-08-21: repeatedly querying/deciding against the same
  held-out set (VisDrone-DET-val, reused across checkpoint selection, mode
  selection, the highres-pilot gate, and tracker/ReID comparisons)
  provably degrades how much a measured result can be trusted to
  generalize, even with no explicit intent to overfit. Their proposed
  fix (`Thresholdout`, a differential-privacy-based reusable holdout) is a
  more principled alternative to this project's current blunt rule
  ("read test-dev exactly once per candidate") -- worth reading before
  designing the next several test-dev reads, since this project will keep
  needing more than one over time.
- **Recht, Roelofs, Schmidt, Shankar, "Do ImageNet Classifiers Generalize
  to ImageNet?"** (ICML 2019,
  [arXiv:1902.10811](https://arxiv.org/abs/1902.10811)). A complementary
  and importantly *different* explanation for the same shape of result:
  rebuilding a fresh ImageNet/CIFAR-10 test set following the original
  collection process, they found consistent accuracy drops (11-14% on
  ImageNet) -- but concluded this was **not** primarily adaptive
  overfitting to the reused test set. Instead, models are simply worse at
  generalizing to a "harder" data distribution, and **gains on the old
  test set did still transfer to the new one at roughly the same rate (no
  diminishing returns)**. This project's own diagnostic (the baseline
  checkpoint's vehicle AP-small also dropped 23% relative from val to
  test-dev, not just the pilot's) is more consistent with the Recht
  framing (test-dev is intrinsically harder) than a pure Dwork-style
  overfitting story -- but the pilot's *relative* gain over baseline still
  shrank far more than Recht's "no diminishing returns" finding would
  predict (val +0.0223 -> test-dev +0.0027, not a constant offset). Both
  papers should be read before concluding which effect dominates here; the
  honest answer right now is "a mix, not fully disentangled."

## Not yet searched, worth a follow-up pass

- Detection transformer / attention-based small-object heads (e.g.
  Deformable DETR variants for aerial imagery), referenced only generically
  by the survey above.
- VisDrone-specific state-of-the-art detector leaderboard entries (what
  AP/AP-small values other published methods actually reach on
  VisDrone-DET-test-dev/-challenge, to calibrate whether this project's
  current numbers are near or far from known achievable performance).
- BoT-SORT's own reported HOTA on MOT17/MOT20 versus VisDrone-MOT, to
  understand how much of the "detection-limited" finding is
  VisDrone-specific versus general to this tracker family.
