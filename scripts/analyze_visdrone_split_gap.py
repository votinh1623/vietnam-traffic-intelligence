"""Compare ground-truth box-scale/class distribution between two VisDrone-DET splits.

Diagnostic only: reads raw VisDrone annotations directly, no detector
inference. Purpose: understand why a detector's AP-small gain measured on
one split does not replicate on another (see docs/benchmark_protocol.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van", "truck",
    "tricycle", "awning-tricycle", "bus", "motor",
]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def parse_annotation_line(line: str) -> tuple[float, float, int] | None:
    parts = [p.strip() for p in line.rstrip(",").split(",")]
    if len(parts) < 6:
        return None
    left, top, width, height = map(float, parts[:4])
    score, category = int(parts[4]), int(parts[5])
    if score != 1 or not 1 <= category <= 10:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height, category


def analyze_split(images_dir: Path, annotations_dir: Path, imgsz: int) -> dict[str, Any]:
    scales: list[float] = []
    scales_by_class: dict[int, list[float]] = {c: [] for c in range(1, 11)}
    image_count = 0
    box_count = 0
    for ann_path in sorted(annotations_dir.glob("*.txt")):
        image_path = images_dir / f"{ann_path.stem}.jpg"
        if not image_path.is_file():
            continue
        with Image.open(image_path) as img:
            width_px, height_px = img.size
        scale_factor = min(imgsz / width_px, imgsz / height_px)
        image_count += 1
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parsed = parse_annotation_line(line)
            if parsed is None:
                continue
            box_w, box_h, category = parsed
            scaled_w = box_w * scale_factor
            scaled_h = box_h * scale_factor
            box_scale = (scaled_w * scaled_h) ** 0.5
            scales.append(box_scale)
            scales_by_class[category].append(box_scale)
            box_count += 1

    scales.sort()
    n = len(scales)
    median = scales[n // 2] if n else float("nan")
    under16 = sum(1 for s in scales if s < 16) / n if n else float("nan")
    under32 = sum(1 for s in scales if s < 32) / n if n else float("nan")

    per_class = {}
    for category, class_scales in scales_by_class.items():
        if not class_scales:
            continue
        class_scales.sort()
        cn = len(class_scales)
        per_class[CLASS_NAMES[category - 1]] = {
            "box_count": cn,
            "median_scale_px": round(class_scales[cn // 2], 2),
            "fraction_under_16px": round(
                sum(1 for s in class_scales if s < 16) / cn, 4
            ),
        }

    return {
        "image_count": image_count,
        "box_count": box_count,
        "boxes_per_image_mean": round(box_count / image_count, 2) if image_count else None,
        "median_scale_px": round(median, 2),
        "fraction_under_16px": round(under16, 4),
        "fraction_under_32px": round(under32, 4),
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--split",
        action="append",
        required=True,
        help="name=images_dir=annotations_dir, repeatable",
    )
    args = parser.parse_args()

    results = {}
    for spec in args.split:
        name, images_dir, annotations_dir = spec.split("=")
        results[name] = analyze_split(resolve(images_dir), resolve(annotations_dir), args.imgsz)

    import json
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
