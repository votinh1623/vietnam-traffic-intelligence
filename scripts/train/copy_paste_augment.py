"""Generate scale-aware copy-paste augmented VisDrone-DET train images.

Offline, deterministic augmentation: crops vehicle-class boxes from
VisDrone2019-DET-train, pastes them (resized into an underrepresented small
scale range) onto a random subset of train images, and writes new
image/label pairs alongside the originals. Parameters are pre-registered
here and must not be tuned after looking at any evaluation result -- see
docs/benchmark_protocol.md's copy-paste augmentation pilot section.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# YOLO-label class indices (0-indexed, matches VisDrone.yaml `names`).
VEHICLE_CLASS_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]  # bicycle..motor, excludes pedestrian/people

# Pre-registered parameters. Do not tune against any evaluation result.
MIN_PASTES_PER_IMAGE = 3
MAX_PASTES_PER_IMAGE = 6
TARGET_SCALE_AT_1280_MIN_PX = 8.0
TARGET_SCALE_AT_1280_MAX_PX = 28.0
LETTERBOX_IMGSZ = 1280
MAX_OVERLAP_IOU = 0.10
MAX_PLACEMENT_ATTEMPTS = 15
MIN_SOURCE_CROP_PX = 6  # skip degenerate near-zero-size source boxes
SEED = 0


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    if not label_path.is_file():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cls, xc, yc, w, h = line.split()
        boxes.append((int(cls), float(xc), float(yc), float(w), float(h)))
    return boxes


def yolo_to_pixel(box: tuple[int, float, float, float, float], img_w: int, img_h: int) -> tuple[int, float, float, float, float]:
    cls, xc, yc, w, h = box
    px_w, px_h = w * img_w, h * img_h
    px_x1 = xc * img_w - px_w / 2
    px_y1 = yc * img_h - px_h / 2
    return cls, px_x1, px_y1, px_w, px_h


def iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class CropBank:
    def __init__(self, train_images_dir: Path, train_labels_dir: Path, rng: random.Random, max_crops_per_class: int = 1500):
        self.crops_by_class: dict[int, list[tuple[Path, tuple[float, float, float, float]]]] = {
            c: [] for c in VEHICLE_CLASS_INDICES
        }
        label_paths = sorted(train_labels_dir.glob("*.txt"))
        rng.shuffle(label_paths)
        for label_path in label_paths:
            if all(len(v) >= max_crops_per_class for v in self.crops_by_class.values()):
                break
            image_path = train_images_dir / f"{label_path.stem}.jpg"
            if not image_path.is_file():
                continue
            img_shape = cv2.imread(str(image_path))
            if img_shape is None:
                continue
            img_h, img_w = img_shape.shape[:2]
            for box in read_yolo_labels(label_path):
                cls = box[0]
                if cls not in VEHICLE_CLASS_INDICES or len(self.crops_by_class[cls]) >= max_crops_per_class:
                    continue
                _, px_x1, px_y1, px_w, px_h = yolo_to_pixel(box, img_w, img_h)
                if px_w < MIN_SOURCE_CROP_PX or px_h < MIN_SOURCE_CROP_PX:
                    continue
                self.crops_by_class[cls].append((image_path, (px_x1, px_y1, px_w, px_h)))

    def sample(self, rng: random.Random) -> tuple[int, np.ndarray] | None:
        available = [c for c, crops in self.crops_by_class.items() if crops]
        if not available:
            return None
        cls = rng.choice(available)
        image_path, (x1, y1, w, h) = rng.choice(self.crops_by_class[cls])
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        x1i, y1i = max(0, int(x1)), max(0, int(y1))
        x2i, y2i = min(img.shape[1], int(x1 + w)), min(img.shape[0], int(y1 + h))
        if x2i <= x1i or y2i <= y1i:
            return None
        return cls, img[y1i:y2i, x1i:x2i].copy()


def augment_image(
    image_path: Path,
    label_path: Path,
    crop_bank: CropBank,
    rng: random.Random,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]] | None:
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    img_h, img_w = img.shape[:2]
    letterbox_scale = min(LETTERBOX_IMGSZ / img_w, LETTERBOX_IMGSZ / img_h)

    existing_boxes = read_yolo_labels(label_path)
    existing_pixel_boxes = [
        (b[1], b[2], b[3], b[4]) for b in (yolo_to_pixel(box, img_w, img_h) for box in existing_boxes)
    ]

    new_labels = list(existing_boxes)
    n_pastes = rng.randint(MIN_PASTES_PER_IMAGE, MAX_PASTES_PER_IMAGE)
    for _ in range(n_pastes):
        sampled = crop_bank.sample(rng)
        if sampled is None:
            continue
        cls, crop = sampled
        crop_h, crop_w = crop.shape[:2]
        if crop_w == 0 or crop_h == 0:
            continue

        target_scale_at_1280 = rng.uniform(TARGET_SCALE_AT_1280_MIN_PX, TARGET_SCALE_AT_1280_MAX_PX)
        target_native_scale = target_scale_at_1280 / letterbox_scale
        aspect = crop_w / crop_h
        new_h = max(3, int(round(target_native_scale / (aspect ** 0.5))))
        new_w = max(3, int(round(target_native_scale * (aspect ** 0.5))))
        if new_w >= img_w or new_h >= img_h:
            continue
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            px = rng.uniform(0, img_w - new_w)
            py = rng.uniform(0, img_h - new_h)
            candidate = (px, py, new_w, new_h)
            if all(iou_xywh(candidate, existing) <= MAX_OVERLAP_IOU for existing in existing_pixel_boxes):
                placed = True
                break
        if not placed:
            continue

        px_i, py_i = int(round(px)), int(round(py))
        img[py_i : py_i + new_h, px_i : px_i + new_w] = resized
        existing_pixel_boxes.append((px, py, new_w, new_h))

        xc = (px + new_w / 2) / img_w
        yc = (py + new_h / 2) / img_h
        nw = new_w / img_w
        nh = new_h / img_h
        new_labels.append((cls, xc, yc, nw, nh))

    return img, new_labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--output-images", required=True)
    parser.add_argument("--output-labels", required=True)
    parser.add_argument("--num-augmented", type=int, required=True)
    parser.add_argument("--suffix", default="_cp")
    args = parser.parse_args()

    rng = random.Random(SEED)
    train_images_dir = resolve(args.train_images)
    train_labels_dir = resolve(args.train_labels)
    output_images_dir = resolve(args.output_images)
    output_labels_dir = resolve(args.output_labels)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    print("Building crop bank...")
    crop_bank = CropBank(train_images_dir, train_labels_dir, rng)
    total_crops = sum(len(v) for v in crop_bank.crops_by_class.values())
    print(f"Crop bank: {total_crops} crops across {len(VEHICLE_CLASS_INDICES)} vehicle classes")

    label_paths = sorted(train_labels_dir.glob("*.txt"))
    rng.shuffle(label_paths)
    selected = label_paths[: args.num_augmented]

    written = 0
    for label_path in selected:
        image_path = train_images_dir / f"{label_path.stem}.jpg"
        if not image_path.is_file():
            continue
        result = augment_image(image_path, label_path, crop_bank, rng)
        if result is None:
            continue
        aug_img, aug_labels = result
        out_stem = f"{label_path.stem}{args.suffix}"
        cv2.imwrite(str(output_images_dir / f"{out_stem}.jpg"), aug_img)
        lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in aug_labels]
        (output_labels_dir / f"{out_stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1
        if written % 500 == 0:
            print(f"{written}/{len(selected)} augmented images written")

    print(f"Done: {written} augmented images written to {output_images_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
