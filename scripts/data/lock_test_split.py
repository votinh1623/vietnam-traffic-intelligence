"""Create an immutable, content-addressed lock manifest for a dataset test split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_lock(dataset_root: Path, manifest_path: Path) -> dict[str, object]:
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    if not rows:
        raise ValueError("manifest contains no test rows")

    items = []
    for row in sorted(rows, key=lambda item: item["image_path"]):
        image_path = dataset_root / row["image_path"]
        label_path = dataset_root / row["label_path"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing test pair: {image_path} / {label_path}")
        items.append(
            {
                "source_id": row["source_id"],
                "frame_index": row["frame_index"],
                "image_path": row["image_path"],
                "label_path": row["label_path"],
                "image_sha256": sha256_file(image_path),
                "label_sha256": sha256_file(label_path),
                "bbox_count": int(row["bbox_count"]),
            }
        )
    canonical = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "status": "locked",
        "locked_on": date.today().isoformat(),
        "policy": "never use this split for training, calibration, prompt selection, model selection, or threshold tuning",
        "image_count": len(items),
        "bbox_count": sum(int(item["bbox_count"]) for item in items),
        "lock_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "items": items,
    }


def verify_lock(dataset_root: Path, lock_path: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    verified_items = []
    for item in lock.get("items", []):
        image_path = dataset_root / item["image_path"]
        label_path = dataset_root / item["label_path"]
        if not image_path.is_file() or not label_path.is_file():
            errors.append(f"missing pair: {image_path} / {label_path}")
            continue
        image_hash = sha256_file(image_path)
        label_hash = sha256_file(label_path)
        if image_hash != item["image_sha256"]:
            errors.append(f"image hash mismatch: {item['image_path']}")
        if label_hash != item["label_sha256"]:
            errors.append(f"label hash mismatch: {item['label_path']}")
        verified_items.append(item)
    canonical = json.dumps(
        verified_items, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    computed_lock_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if computed_lock_hash != lock.get("lock_sha256"):
        errors.append("lock manifest digest mismatch")
    if len(verified_items) != lock.get("image_count"):
        errors.append("test image count mismatch")
    return {
        "valid": not errors,
        "expected_lock_sha256": lock.get("lock_sha256"),
        "computed_lock_sha256": computed_lock_hash,
        "verified_image_count": len(verified_items),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"lock already exists: {output}")
    lock = build_lock(args.dataset_root.resolve(), args.manifest.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in lock.items() if key != "items"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
