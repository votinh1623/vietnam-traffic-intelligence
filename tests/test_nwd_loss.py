from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "train"))

try:
    import torch

    from nwd_loss import NWDBboxLoss, patch_bbox_loss, wasserstein_similarity
    from ultralytics.utils.loss import BboxLoss

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _forward_inputs(reg_max: int = 16, offset: float = 0.0):
    """Build minimal but shape-faithful inputs for BboxLoss/NWDBboxLoss.forward:
    1 image, 4 anchors, 2 of them foreground, 1 class."""
    anchor_points = torch.tensor(
        [[5.0, 5.0], [5.0, 5.0], [50.0, 50.0], [50.0, 50.0]]
    )
    target_bboxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0], [40.0, 40.0, 60.0, 60.0], [40.0, 40.0, 60.0, 60.0]]]
    )
    pred_bboxes = (target_bboxes + offset).clone().requires_grad_(True)
    fg_mask = torch.tensor([[True, False, True, False]])
    target_scores = torch.zeros((1, 4, 1))
    target_scores[0, 0, 0] = 0.9
    target_scores[0, 2, 0] = 0.8
    target_scores_sum = target_scores.sum().clamp(min=1e-7)
    reg_max_bins = reg_max
    pred_dist = torch.rand((1, 4, 4 * reg_max_bins), requires_grad=True)
    imgsz = torch.tensor([640.0, 640.0])
    stride = torch.tensor([8.0])
    return dict(
        pred_dist=pred_dist,
        pred_bboxes=pred_bboxes,
        anchor_points=anchor_points,
        target_bboxes=target_bboxes,
        target_scores=target_scores,
        target_scores_sum=target_scores_sum,
        fg_mask=fg_mask,
        imgsz=imgsz,
        stride=stride,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch/ultralytics not installed")
class WassersteinSimilarityTests(unittest.TestCase):
    def test_identical_boxes_score_one(self) -> None:
        # A small eps inside the sqrt (for gradient stability at distance 0)
        # keeps this just under 1.0; verify it's within that eps, not exact.
        box = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        similarity = wasserstein_similarity(box, box, constant=12.8)
        self.assertGreater(similarity.item(), 0.9999)

    def test_similarity_decreases_with_offset(self) -> None:
        box_a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        box_b = torch.tensor([[2.0, 0.0, 12.0, 10.0]])
        near = wasserstein_similarity(box_a, box_a, constant=12.8).item()
        far = wasserstein_similarity(box_a, box_b, constant=12.8).item()
        self.assertGreater(near, far)

    def test_degrades_more_gracefully_than_iou_for_tiny_boxes(self) -> None:
        # A 2px shift on a 4x4 box collapses IoU close to zero, but this is
        # exactly the failure mode NWD is meant to avoid: it should retain
        # much more of its similarity score for the same absolute shift.
        from ultralytics.utils.metrics import bbox_iou

        tiny_a = torch.tensor([[0.0, 0.0, 4.0, 4.0]])
        tiny_b = torch.tensor([[2.0, 0.0, 6.0, 4.0]])
        iou = bbox_iou(tiny_a, tiny_b, xywh=False).item()
        nwd = wasserstein_similarity(tiny_a, tiny_b, constant=12.8).item()
        self.assertGreater(nwd, iou)


@unittest.skipUnless(TORCH_AVAILABLE, "torch/ultralytics not installed")
class NWDBboxLossTests(unittest.TestCase):
    def test_alpha_zero_matches_original_ciou_loss(self) -> None:
        """alpha=0 must reduce the blend to pure CIoU -- i.e. numerically
        match the untouched, original BboxLoss.forward on the same inputs.
        This is the main correctness check: it does not require re-deriving
        the DFL math, only that alpha=0 is truly a no-op wrapper around it."""
        torch.manual_seed(0)
        inputs = _forward_inputs()
        original = BboxLoss(reg_max=16)
        patched = NWDBboxLoss(reg_max=16)
        patched.nwd_alpha = 0.0
        patched.nwd_constant = 12.8

        expected_iou, expected_dfl = original.forward(**{k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()})
        actual_iou, actual_dfl = patched.forward(**{k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()})

        self.assertAlmostEqual(expected_iou.item(), actual_iou.item(), places=5)
        self.assertAlmostEqual(expected_dfl.item(), actual_dfl.item(), places=5)

    def test_perfect_prediction_has_near_zero_iou_loss(self) -> None:
        inputs = _forward_inputs(offset=0.0)
        loss = NWDBboxLoss(reg_max=16)
        loss.nwd_alpha = 0.5
        loss_iou, _ = loss.forward(**inputs)
        self.assertLess(loss_iou.item(), 1e-3)

    def test_gradient_flows_to_predicted_boxes(self) -> None:
        inputs = _forward_inputs(offset=3.0)
        loss = NWDBboxLoss(reg_max=16)
        loss.nwd_alpha = 0.5
        loss_iou, loss_dfl = loss.forward(**inputs)
        (loss_iou + loss_dfl).backward()
        self.assertIsNotNone(inputs["pred_bboxes"].grad)
        self.assertTrue(torch.isfinite(inputs["pred_bboxes"].grad).all())

    def test_alpha_changes_the_loss_value(self) -> None:
        inputs_a = _forward_inputs(offset=3.0)
        inputs_b = {k: v.clone().detach().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v for k, v in inputs_a.items()}
        loss_ciou_only = NWDBboxLoss(reg_max=16)
        loss_ciou_only.nwd_alpha = 0.0
        loss_nwd_heavy = NWDBboxLoss(reg_max=16)
        loss_nwd_heavy.nwd_alpha = 1.0

        iou_loss_value, _ = loss_ciou_only.forward(**inputs_a)
        nwd_loss_value, _ = loss_nwd_heavy.forward(**inputs_b)
        self.assertNotAlmostEqual(iou_loss_value.item(), nwd_loss_value.item(), places=4)

    def test_patch_bbox_loss_replaces_module_attribute(self) -> None:
        import ultralytics.utils.loss as loss_module

        original = loss_module.BboxLoss
        try:
            patch_bbox_loss(alpha=0.3, constant=10.0)
            self.assertIs(loss_module.BboxLoss, NWDBboxLoss)
            self.assertEqual(NWDBboxLoss.nwd_alpha, 0.3)
            self.assertEqual(NWDBboxLoss.nwd_constant, 10.0)
        finally:
            loss_module.BboxLoss = original


if __name__ == "__main__":
    unittest.main()
