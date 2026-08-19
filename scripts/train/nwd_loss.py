"""Normalized Wasserstein Distance (NWD) bbox loss for YOLOv8, monkey-patched
into Ultralytics' loss module.

Rationale (see readme.md "Detection" results / benchmark_protocol.md
"Detector training and validation"): the locked-test generalization gap is
driven by an object-scale shift -- 82.9-100% of locked-test boxes cover
under 0.1% of the image, by class. IoU-based loss (Ultralytics' default
CIoU) is numerically unstable for boxes this small: a 1-2 pixel localization
error can collapse IoU from ~0.5 to ~0.1, so the loss landscape is noisy
exactly where it matters most for this dataset.

NWD models each box as a 2D Gaussian (center = box center, diagonal
covariance = (w/2, h/2)) and uses the closed-form Wasserstein-2 distance
between two such Gaussians, then maps it to a bounded [0, 1] similarity via
exp(-sqrt(W2)/C). Unlike IoU, this stays smooth and non-zero even for
non-overlapping tiny boxes, which is the known motivation from "A Normalized
Gaussian Wasserstein Distance for Tiny Object Detection" (Wang et al.).

`nwd_constant` (C) is a scale hyperparameter, not a fitted value: the paper's
own reference value (12.8, tuned for AI-TOD) is used as the default here and
has NOT been retuned to this project's own box-size distribution -- see
patch_bbox_loss()'s docstring before trusting a specific constant.
"""

from __future__ import annotations

import torch
from torch import nn

from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


def wasserstein_similarity(
    box1_xyxy: torch.Tensor, box2_xyxy: torch.Tensor, constant: float, eps: float = 1e-7
) -> torch.Tensor:
    """Elementwise NWD similarity in [0, 1] between paired xyxy boxes.

    Both tensors must already be paired (box1_xyxy[i] compared with
    box2_xyxy[i]), matching how BboxLoss.forward calls bbox_iou -- this is
    not a full pairwise (N, M) matrix.
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy.unbind(-1)
    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)
    cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    center_term = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    size_term = ((w1 - w2) / 2) ** 2 + ((h1 - h2) / 2) ** 2
    wasserstein_2 = center_term + size_term
    return torch.exp(-torch.sqrt(wasserstein_2 + eps) / constant)


class NWDBboxLoss(BboxLoss):
    """BboxLoss with the CIoU term blended with NWD similarity.

    `nwd_alpha` and `nwd_constant` are class attributes (not constructor
    arguments) because Ultralytics' v8DetectionLoss constructs this class
    with a single positional arg (`BboxLoss(m.reg_max)`); patch_bbox_loss()
    sets them on the class before training starts.
    """

    nwd_alpha: float = 0.5
    nwd_constant: float = 12.8

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        nwd = wasserstein_similarity(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], constant=self.nwd_constant
        ).unsqueeze(-1)
        similarity = self.nwd_alpha * nwd + (1.0 - self.nwd_alpha) * iou
        loss_iou = ((1.0 - similarity) * weight).sum() / target_scores_sum

        # DFL loss: unchanged from the parent implementation.
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                nn.functional.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(
                    -1, keepdim=True
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


def patch_bbox_loss(alpha: float = 0.5, constant: float = 12.8) -> None:
    """Monkey-patch ultralytics.utils.loss.BboxLoss with NWDBboxLoss.

    Must be called before model.train() / v8DetectionLoss.__init__ runs
    (that is where `self.bbox_loss = BboxLoss(m.reg_max)` is constructed),
    since v8DetectionLoss looks up the bare name `BboxLoss` from its own
    module's globals at call time -- patching that module attribute here is
    picked up even though v8DetectionLoss.py itself is never edited.

    `constant` (C in the NWD paper) should ideally be set from this
    project's own box-size distribution (e.g. the mean of sqrt(w*h) over
    Vietnam v5 train boxes) rather than left at the AI-TOD-tuned default of
    12.8; treat any run using the default as an unvalidated starting point,
    not a tuned choice.
    """
    import ultralytics.utils.loss as loss_module

    NWDBboxLoss.nwd_alpha = alpha
    NWDBboxLoss.nwd_constant = constant
    loss_module.BboxLoss = NWDBboxLoss
