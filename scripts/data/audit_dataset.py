"""Build a non-destructive dataset manifest and audit split leakage.

The tool only reads images/labels and writes small CSV/JSON reports. It never
moves, copies, renames, or edits dataset content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FRAME_RE = re.compile(
    r"^(?P<source>.+)_frame_(?P<frame>\d+)(?:_jpg)?(?:\.rf\.[0-9a-f]+)?$",
    re.IGNORECASE,
)


def discover_split_dirs(dataset_root: Path) -> dict[str, str]:
    """Support both the legacy Roboflow names and the normalized v4 names."""
    result: dict[str, str] = {}
    if (dataset_root / "train").is_dir():
        result["train"] = "train"
    if (dataset_root / "calibration").is_dir():
        result["calibration"] = "calibration"
    if (dataset_root / "validation").is_dir():
        result["validation"] = "validation"
    elif (dataset_root / "valid").is_dir():
        result["validation"] = "valid"
    if (dataset_root / "test").is_dir():
        result["test"] = "test"
    return result


def parse_source_frame(filename: str) -> tuple[str, int | None]:
    """Recover source video and extracted-frame index from an exported name."""
    match = FRAME_RE.match(Path(filename).stem)
    if not match:
        return "unknown", None
    return match.group("source"), int(match.group("frame"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_label(path: Path) -> tuple[str, int, str, str]:
    if not path.exists():
        return "missing", 0, "", ""
    class_counts: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    malformed = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if not parts[0].isdigit():
                malformed += 1
                continue
            if len(parts) == 5:
                formats["bbox"] += 1
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                formats["polygon"] += 1
            else:
                malformed += 1
                continue
            class_counts[parts[0]] += 1
    box_count = sum(class_counts.values())
    status = "empty" if box_count == 0 and malformed == 0 else "labeled"
    if malformed:
        status = "malformed"
    counts = ";".join(f"{key}:{class_counts[key]}" for key in sorted(class_counts))
    annotation_formats = ";".join(
        f"{key}:{formats[key]}" for key in sorted(formats)
    )
    return status, box_count, counts, annotation_formats


def build_manifest(
    dataset_root: Path,
    frame_interval_seconds: float = 0.5,
    hash_content: bool = True,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, directory_name in discover_split_dirs(dataset_root).items():
        image_dir = dataset_root / directory_name / "images"
        label_dir = dataset_root / directory_name / "labels"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir(), key=lambda item: item.name.lower()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            source_id, frame_index = parse_source_frame(image_path.name)
            label_path = label_dir / f"{image_path.stem}.txt"
            label_status, box_count, class_counts, annotation_formats = inspect_label(
                label_path
            )
            image_hash = sha256_file(image_path) if hash_content else ""
            image_id = image_hash or hashlib.sha256(
                image_path.relative_to(dataset_root).as_posix().encode("utf-8")
            ).hexdigest()
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path.relative_to(dataset_root).as_posix(),
                    "label_path": (
                        label_path.relative_to(dataset_root).as_posix()
                        if label_path.exists()
                        else ""
                    ),
                    "source_id": source_id,
                    "frame_index": "" if frame_index is None else frame_index,
                    "timestamp_seconds": (
                        "" if frame_index is None else round(frame_index * frame_interval_seconds, 3)
                    ),
                    "split": split,
                    "label_status": label_status,
                    "box_count": box_count,
                    "class_counts": class_counts,
                    "annotation_formats": annotation_formats,
                    "content_sha256": image_hash,
                }
            )
    return rows


def _temporal_gap_summary(
    rows: Iterable[dict[str, object]], candidate_split: str
) -> dict[str, object]:
    train_frames: dict[str, list[int]] = defaultdict(list)
    candidates: list[tuple[str, int]] = []
    for row in rows:
        frame = row["frame_index"]
        if frame == "" or row["source_id"] == "unknown":
            continue
        item = (str(row["source_id"]), int(frame))
        if row["split"] == "train":
            train_frames[item[0]].append(item[1])
        elif row["split"] == candidate_split:
            candidates.append(item)

    distances = []
    for source_id, frame in candidates:
        if train_frames[source_id]:
            distances.append(min(abs(frame - train) for train in train_frames[source_id]))
    ordered = sorted(distances)
    return {
        "parsed_frames": len(distances),
        "gap_le_1": sum(distance <= 1 for distance in distances),
        "gap_le_2": sum(distance <= 2 for distance in distances),
        "gap_le_5": sum(distance <= 5 for distance in distances),
        "minimum": ordered[0] if ordered else None,
        "median": ordered[len(ordered) // 2] if ordered else None,
        "maximum": ordered[-1] if ordered else None,
    }


def audit_manifest(rows: list[dict[str, object]]) -> dict[str, object]:
    source_splits: dict[str, set[str]] = defaultdict(set)
    source_frame_splits: dict[tuple[str, int], set[str]] = defaultdict(set)
    hash_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    label_status: Counter[str] = Counter()
    annotation_formats: Counter[str] = Counter()

    for row in rows:
        split = str(row["split"])
        source_id = str(row["source_id"])
        split_counts[split] += 1
        label_status[str(row["label_status"])] += 1
        for item in str(row["annotation_formats"]).split(";"):
            if not item:
                continue
            name, count = item.split(":", 1)
            annotation_formats[name] += int(count)
        if source_id != "unknown":
            source_splits[source_id].add(split)
            if row["frame_index"] != "":
                source_frame_splits[(source_id, int(row["frame_index"]))].add(split)
        if row["content_sha256"]:
            hash_splits[str(row["content_sha256"])].add(split)

    overlapping_sources = {
        source: sorted(splits)
        for source, splits in source_splits.items()
        if len(splits) > 1
    }
    same_frame_groups = [
        {"source_id": source, "frame_index": frame, "splits": sorted(splits)}
        for (source, frame), splits in source_frame_splits.items()
        if len(splits) > 1
    ]
    duplicate_hashes = [
        {"content_sha256": digest, "splits": sorted(splits)}
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    ]
    return {
        "status": "invalid_leakage" if overlapping_sources else "no_group_leakage_detected",
        "image_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "label_status_counts": dict(sorted(label_status.items())),
        "annotation_format_counts": dict(sorted(annotation_formats.items())),
        "unknown_source_images": sum(row["source_id"] == "unknown" for row in rows),
        "overlapping_source_count": len(overlapping_sources),
        "overlapping_sources": overlapping_sources,
        "same_source_frame_cross_split_count": len(same_frame_groups),
        "same_source_frame_cross_split": same_frame_groups,
        "exact_content_hash_cross_split_count": len(duplicate_hashes),
        "exact_content_hash_cross_split": duplicate_hashes,
        "validation_vs_train_temporal_gap_frames": _temporal_gap_summary(rows, "validation"),
        "test_vs_train_temporal_gap_frames": _temporal_gap_summary(rows, "test"),
    }


def propose_source_split(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Create a deterministic source-only proposal without moving any files."""
    source_keys: dict[str, set[tuple[str, object]]] = defaultdict(set)
    for row in rows:
        source = str(row["source_id"])
        if source == "unknown":
            continue
        key: object = row["frame_index"] if row["frame_index"] != "" else row["image_id"]
        source_keys[source].add((source, key))

    split_order = ("train", "calibration", "validation", "test")
    ratios = {"train": 0.65, "calibration": 0.10, "validation": 0.10, "test": 0.15}
    total = sum(len(keys) for keys in source_keys.values())
    assigned = {split: 0 for split in split_order}
    result: list[dict[str, object]] = []

    ordered_sources = sorted(source_keys, key=lambda source: (-len(source_keys[source]), source))
    for source in ordered_sources:
        count = len(source_keys[source])
        destination = max(
            split_order,
            key=lambda split: ratios[split] * total - assigned[split],
        )
        assigned[destination] += count
        result.append(
            {
                "source_id": source,
                "unique_frame_count": count,
                "proposed_split": destination,
                "status": "proposal_requires_review",
            }
        )
    return sorted(result, key=lambda row: str(row["source_id"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--skip-content-hash", action="store_true")
    parser.add_argument("--fail-on-leakage", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        parser.error(f"Dataset root does not exist: {dataset_root}")
    rows = build_manifest(
        dataset_root,
        frame_interval_seconds=args.frame_interval,
        hash_content=not args.skip_content_hash,
    )
    if not rows:
        parser.error(f"No images found under: {dataset_root}")

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "manifest.csv", rows)
    write_csv(output_dir / "source_split_proposal.csv", propose_source_split(rows))
    audit = audit_manifest(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.fail_on_leakage and audit["status"] == "invalid_leakage":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
