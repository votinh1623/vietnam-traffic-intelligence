import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "train"))

from copy_paste_augment import (  # noqa: E402
    CropBank,
    augment_image,
    iou_xywh,
    read_yolo_labels,
    yolo_to_pixel,
)


class TestIouXywh(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        box = (10.0, 10.0, 20.0, 20.0)
        self.assertAlmostEqual(iou_xywh(box, box), 1.0)

    def test_disjoint_boxes_have_iou_zero(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (100.0, 100.0, 10.0, 10.0)
        self.assertEqual(iou_xywh(a, b), 0.0)

    def test_half_overlap_is_between_zero_and_one(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 10.0, 10.0)
        iou = iou_xywh(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


class TestYoloConversion(unittest.TestCase):
    def test_yolo_to_pixel_centers_box_correctly(self):
        box = (3, 0.5, 0.5, 0.2, 0.2)
        cls, x1, y1, w, h = yolo_to_pixel(box, img_w=100, img_h=100)
        self.assertEqual(cls, 3)
        self.assertAlmostEqual(x1, 40.0)
        self.assertAlmostEqual(y1, 40.0)
        self.assertAlmostEqual(w, 20.0)
        self.assertAlmostEqual(h, 20.0)

    def test_read_yolo_labels_round_trip(self, tmp_path=None):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            label_path = Path(tmp) / "0001.txt"
            label_path.write_text("3 0.5 0.5 0.2 0.2\n9 0.1 0.1 0.05 0.05\n", encoding="utf-8")
            boxes = read_yolo_labels(label_path)
            self.assertEqual(len(boxes), 2)
            self.assertEqual(boxes[0], (3, 0.5, 0.5, 0.2, 0.2))
            self.assertEqual(boxes[1], (9, 0.1, 0.1, 0.05, 0.05))

    def test_missing_label_file_returns_empty(self):
        self.assertEqual(read_yolo_labels(Path("does-not-exist.txt")), [])


class TestAugmentImage(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.images_dir.mkdir()
        self.labels_dir.mkdir()

        # A source image with one car box, used both as the crop-bank source
        # and as the image being augmented.
        img = np.full((400, 400, 3), 128, dtype=np.uint8)
        cv2.rectangle(img, (150, 150), (250, 250), (0, 0, 255), -1)
        cv2.imwrite(str(self.images_dir / "0001.jpg"), img)
        (self.labels_dir / "0001.txt").write_text("3 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_augment_image_adds_new_small_boxes_without_removing_existing(self):
        import random

        rng = random.Random(0)
        crop_bank = CropBank(self.images_dir, self.labels_dir, rng, max_crops_per_class=10)
        total_crops = sum(len(v) for v in crop_bank.crops_by_class.values())
        self.assertGreater(total_crops, 0, "crop bank should find the car box in the fixture image")

        result = augment_image(
            self.images_dir / "0001.jpg",
            self.labels_dir / "0001.txt",
            crop_bank,
            rng,
        )
        self.assertIsNotNone(result)
        aug_img, aug_labels = result
        self.assertEqual(aug_img.shape[:2], (400, 400))

        original_boxes = read_yolo_labels(self.labels_dir / "0001.txt")
        self.assertEqual(aug_labels[: len(original_boxes)], original_boxes)
        self.assertGreaterEqual(len(aug_labels), len(original_boxes))

        for cls, xc, yc, w, h in aug_labels:
            self.assertTrue(0.0 <= xc <= 1.0)
            self.assertTrue(0.0 <= yc <= 1.0)
            self.assertGreater(w, 0.0)
            self.assertGreater(h, 0.0)

    def test_pasted_boxes_do_not_overlap_existing_beyond_threshold(self):
        import random

        from copy_paste_augment import MAX_OVERLAP_IOU

        rng = random.Random(0)
        crop_bank = CropBank(self.images_dir, self.labels_dir, rng, max_crops_per_class=10)
        result = augment_image(
            self.images_dir / "0001.jpg",
            self.labels_dir / "0001.txt",
            crop_bank,
            rng,
        )
        self.assertIsNotNone(result)
        _, aug_labels = result
        original_box = yolo_to_pixel(aug_labels[0], img_w=400, img_h=400)
        original_pixel = (original_box[1], original_box[2], original_box[3], original_box[4])

        for box in aug_labels[1:]:
            pixel_box = yolo_to_pixel(box, img_w=400, img_h=400)
            candidate = (pixel_box[1], pixel_box[2], pixel_box[3], pixel_box[4])
            self.assertLessEqual(iou_xywh(candidate, original_pixel), MAX_OVERLAP_IOU + 1e-6)


if __name__ == "__main__":
    unittest.main()
